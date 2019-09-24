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

class PlanB:
    box_score_threshold = 0.8
    move_period = 10
    move_duration = 3

    def __init__(self):
        self.upstream = Upstream()
        self.target = None
        self.start_time = time.time()

    def process_boxes(self, image, meta, boxes, scores):
        # called after boxes are detected
        logger.debug('process_boxes(): boxes=%d, target locked=%s' % (len(boxes), (self.target is not None)))
        if self.target is None:
            self.process_boxes_for_new_target(image, meta, boxes, scores)
        else:
            self.process_boxes_for_locked_target(image, meta, boxes, scores)

    # ============================================================================

    import time

    def reset_move(self):
        self.start_time = time.time()

    def is_camera_moving(self):
        elapsed = (time.time() - self.start_time) % self.move_period
        return (elapsed < self.move_duration)

    # ============================================================================

    def process_boxes_for_new_target(self, image, meta, boxes, scores):
        # 5
        [i, sq] = self.find_best_target(image, meta, boxes, scores)
        if i is None:
            logger.info('process_boxes_for_new_target(): face not found')
            self.upstream.send_event_no_target_lock(image, meta)    
        else:
            box = boxes[i]
            logger.info('process_boxes_for_new_target(): best box=%s, score=%s, sq=%f' % (box, scores[i], sq))
            self.acquire_target(image, meta, box)
            self.upstream.send_event_target_locked(image, meta, box)

    def process_boxes_for_locked_target(self, image, meta, boxes, scores):
        # 11
        [i, dist] = self.find_locked_target(image, meta, boxes, scores)
        if i is None:
            logger.info('process_boxes_for_locked_target(): face not found')
            self.upstream.send_event_target_lost(image, meta)    
        else:
            box = boxes[i]
            logger.info('process_boxes_for_locked_target(): best box=%s, score=%s, dist=%f' % (box, scores[i], dist))
            smoothed_box = self.update_target(image, meta, box)
            self.upstream.send_event_target_locked(image, meta, smoothed_box)

    # ----------------------------------------------------------------------------

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
        return [closest_box_i, closest_box_distance]

    def acquire_target(self, image, meta, box):
        # 9 - called once after target is selected
        self.target = { 'box': box }

    def update_target(self, image, meta, box):
        # 14 - called continuously after target is
        self.target = { 'box': box }
        return box

    def release_target(self):
        # invoked when camera starts moving
        self.target = None

    # ====================================================================================================

    def show_info(self, frame):
        # Put status attributes on the image
        def putText(row, text):
            scale = 0.5
            cv2.putText(frame, text, (0, int(25*row*scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, 255)
        
        elapsed = (time.time() - self.start_time) % self.move_period
        putText(1, 'elapsed=%.2f' % elapsed)
