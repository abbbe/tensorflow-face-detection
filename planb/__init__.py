import cv2
import math
import time

from planb.upstream import Upstream

import logging
logger = logging.getLogger('planb')

def _get_box_distance(box1, box2):
    # return distance between the centers of the boxes
    box1_cx = (box1[0] + box1[2])/2
    box1_cy = (box1[1] + box1[3])/2
    box2_cx = (box2[0] + box2[2])/2
    box2_cy = (box2[1] + box2[3])/2
    dx = box1_cx - box2_cx
    dy = box1_cy - box2_cy
    return math.sqrt(dx*dx + dy*dy)

def _get_box_sq(box):
    # return distance between the centers of the boxes
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    return math.fabs(box_w * box_h)

# ========================================================================================================

from enum import Enum
class Mode(Enum):
    MOVING = 1
    IDLE = 2
    EFFECT_START = 3
    EFFECT_RUN = 4
    EFFECT_ABORT = 5

class PlanB:
    mode = Mode(Mode.IDLE)
    box_score_threshold = 0.8
    move_period = 10
    move_duration = 3

    T1 = 3
    T2 = 10
    T3 = 3

    def __init__(self):
        self.upstream = Upstream()
        self.target = None
        self.start_time = time.time()

    # ============================================================================

    import time

    def reset_move(self):
        self.start_time = time.time()

    def run(self, image, meta, boxes, scores):
        # how many seconds passed since the start of the cycle
        elapsed = (time.time() - self.start_time) % self.move_period
        moving = (elapsed < self.move_duration)

        # process single frame
        # send messages upstream

        if self.mode == Mode.MOVING:
            if moving:
                # still moving
                return # end of processing
            else:
                self.mode = Mode.IDLE
                # fall through

        if self.mode == Mode.IDLE:
            if moving:
                # started to move
                self.mode = Mode.MOVING
                return # end of processing

            [i, _] = self.find_best_target(image, meta, boxes, scores)
            if i is None:
                return # end of processing
            
            # new face found
            self.mode = Mode.EFFECT_START
            self.mode_endtime = time.time() + self.T1
            self.upstream.send_effect_start(image, meta, boxes[i], [self.T1, self.T2, self.T3])
            return # end of processing

        if self.mode == Mode.EFFECT_START:
            if time.time() < self.mode_endtime:
                [i, _] = self.find_locked_target(image, meta, boxes, scores)
                if i is None:
                    self.mode = Mode.EFFECT_ABORT
                    self.mode_endtime = time.time() + self.T3
                    self.upstream.send_effect_abort(image, meta, [self.T3]) 
                    return # end of processing

                box = boxes[i]
                self.upstream.send_effect_run(image, meta, box)
                return # end of processing
                
            self.mode = Mode.EFFECT_RUN
            self.mode_endtime = time.time() + self.T2
            # fall through
        
        if self.mode == Mode.EFFECT_RUN:
            if time.time() < self.mode_endtime:
                return # end of processing

            self.mode = Mode.EFFECT_ABORT
            self.mode_endtime = time.time() + self.T3
            return # end of processing

        if self.mode == Mode.EFFECT_ABORT:
            if time.time() < self.mode_endtime:
                return # end of processing

            if moving:
                self.mode = Mode.MOVING
            else:
                self.mode = Mode.IDLE
            return # end of processing

    # ============================================================================

    def find_best_target(self, image, meta, boxes, scores):
        # FIXME-6: find the biggest face
        best_box_i = None
        best_box_sq = 0

        for i in range(len(boxes)):
            if scores[i] < self.box_score_threshold:
                # ignore low-certainty faces
                continue
    
            box = boxes[i]
            sq = _get_box_sq(box)
            if sq > best_box_sq:
                best_box_i = i
                best_box_sq = sq

        if best_box_i is not None:
            self.target = { 'box': boxes[best_box_i] }

        return [best_box_i, best_box_sq]

    def find_locked_target(self, image, meta, boxes, scores):
        # FIXME-12: find box closest to the target
        closest_box_i = None
        closest_box_distance = float('inf')

        for i in range(len(boxes)):
            if scores[i] < self.box_score_threshold:
                # ignore low-certainty faces
                continue
    
            box = boxes[i]
            distance = _get_box_distance(self.target['box'], box)
            if distance < closest_box_distance:
                closest_box_i = i
                closest_box_distance = distance

        if closest_box_i is not None:
            self.target = { 'box': boxes[closest_box_i] }

        return [closest_box_i, closest_box_distance]

    # ====================================================================================================

    def show_info(self, frame):
        # Put status attributes on the image
        def putText(row, text):
            scale = 0.5
            cv2.putText(frame, text, (0, int(25*row*scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, 128)
        
        elapsed = (time.time() - self.start_time) % self.move_period
        putText(1, 'elapsed=%.2f' % elapsed)
        putText(2, 'mode=%s' % self.mode)
        if self.mode == Mode.EFFECT_START or self.mode == Mode.EFFECT_RUN or self.mode == Mode.EFFECT_ABORT:
            putText(3, 'mode_remains=%.2f' % (self.mode_endtime - time.time()))
