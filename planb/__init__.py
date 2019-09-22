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
import json
import base64

clientsocket = None
fileh = None

def connect_upstream(port, file):
    global clientsocket, fileh
    if port >= 0:
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.connect(('localhost', port))
        logger.info('Connected upstream')
    if file is not None:
        fileh = open(file, 'wb')

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
    send_event(image, meta, { 'camera_moves': False, 'target_locked': True, 'target_box': box.tolist() })
    return

def send_event_target_lost(image, meta):
    # 15
    send_event(image, meta, { 'camera_moves': False, 'target_locked': True })
    return

def send_event(image, meta, event):
    logger.info('send_event(%s)' % event)

    if clientsocket is not None or fileh is not None:
        # add frame to the event structure
        image = cv2.resize(image, (64, 48)) # FIXME
        _, imdata = cv2.imencode('.jpg', image)
        encoded_imdata = base64.b64encode(imdata).decode('ascii')
        event['frame'] = encoded_imdata 

        msg = (json.dumps(event) + '\r\n').encode('ascii')

        if clientsocket is not None:
            clientsocket.sendall(msg)
        if fileh is not None:
            fileh.write(msg)

def deinit():
    if clientsocket is not None:
        clientsocket.close()
        clientsocket = None
    if fileh is not None:
        fileh.close()
        fileh = None
