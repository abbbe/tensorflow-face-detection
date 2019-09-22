import cv2
import math

import logging
logger = logging.getLogger('planb')
logging.basicConfig(level=logging.ERROR)

target = None
box_score_threshold = 0.8

def process_image(image, meta):
    # called on every captured frame
    # if returns true, higher-level code will do face detection and call process_boxes()

    if is_camera_moving(image, meta):
        # camera is moving: inform Unity
        send_event_camera_moving(image, meta)
        # forget about the target
        target = None
        return False
    else:
        # return True, indicating higher-level code they have to do face detection and call process_boxes()
        return True

def process_boxes(image, meta, boxes, scores):
    # called after boxes are detected
    logger.debug('process_boxes(): boxes=%d, target locked=%s' % (len(boxes), (target is not None)))
    if target is None:
        process_boxes_for_new_target(image, meta, boxes, scores)
    else:
        process_boxes_for_locked_target(image, meta, boxes, scores)

# ============================================================================

def is_camera_moving(image, meta):
    # Compare optical flow between the given image and previous one,
    # tell between some object moving vs all object drifting
    # Return True if we think camera is moving
    # FIXME-1: implement camera movement detection
    return False

# ============================================================================

def process_boxes_for_new_target(image, meta, boxes, scores):
    # 5
    [i, sq] = find_best_target(image, meta, boxes, scores)
    if i is None:
        logger.info('process_boxes_for_new_target(): face not found')
        send_event_no_target_lock(image, meta)    
    else:
        box = boxes[i]
        logger.info('process_boxes_for_new_target(): best box=%s, score=%s, sq=%f' % (box, scores[i], sq))
        acquire_target(image, meta, box)
        send_event_target_locked(image, meta, box)

def process_boxes_for_locked_target(image, meta, boxes, scores):
    # 11
    [i, dist] = find_locked_target(image, meta, boxes, scores, target)
    if i is None:
        logger.info('process_boxes_for_locked_target(): face not found')
        send_event_target_lost(image, meta)    
    else:
        box = boxes[i]
        logger.info('process_boxes_for_locked_target(): best box=%s, score=%s, dist=%f' % (box, scores[i], dist))
        update_target(image, meta, box)
        send_event_target_locked(image, meta, box)

# ----------------------------------------------------------------------------

def find_best_target(image, meta, boxes, scores):
    # FIXME-6: find the biggest face
    best_box_i = None
    best_box_sq = 0

    for i in range(len(boxes)):
        if scores[i] < box_score_threshold:
            # ignore low-certainty faces
            continue
    
        box = boxes[i]
        sq = _get_box_sq(box)
        if sq > best_box_sq:
            best_box_i = i
            best_box_sq = sq
    return [best_box_i, best_box_sq]

def find_locked_target(image, meta, boxes, scores, target):
    # FIXME-12: find box closest to the target
    closest_box_i = None
    closest_box_distance = float('inf')

    for i in range(len(boxes)):
        if scores[i] < box_score_threshold:
            # ignore low-certainty faces
            continue
    
        box = boxes[i]
        distance = _get_box_distance(target['box'], box)
        if distance < closest_box_distance:
            closest_box_i = i
            closest_box_distance = distance
    return [closest_box_i, closest_box_distance]

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

def acquire_target(image, meta, box):
    # 9
    global target
    target = { 'box': box }

def update_target(image, meta, box):
    acquire_target(image, meta, box)

# ============================================================================

import socket
import struct
import pickle

HOST = ''
PORT = 8089 # or None to disable upstream connection
clientsocket = None

def connect_upstream():
    global clientsocket
    if PORT is not None:
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.connect(('localhost', 8089))
        logger.info('Connected upstream')

def send_event_camera_moving(image, meta):
    # 3
    send_event(image, meta, { 'camera_moves': True })
    return

def send_event_no_target_lock(image, meta):
    # 8
    send_event(image, meta, { 'camera_moves': False, 'target_locked': False })
    return

def send_event_target_locked(image, meta, box):
    # 10
    send_event(image, meta, { 'camera_moves': False, 'target_locked': True, 'target_box': box })
    return

def send_event_target_lost(image, meta):
    # 15
    send_event(image, meta, { 'camera_moves': False, 'target_locked': True })
    return

def send_event(image, meta, event):
    logger.info('send_event(%s)' % event)

    if clientsocket is not None:
        # convert back to BGR and resize
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #image = cv2.resize(image, (640, 480))  # resize the frame        

        # serialize metadata, event, and image and send upstream
        data = pickle.dumps([meta, event, image])
        message_size = struct.pack("L", len(data)) ### CHANGED
        clientsocket.sendall(message_size + data)
