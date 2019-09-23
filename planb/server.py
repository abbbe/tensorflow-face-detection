import pickle
import socket
import json
import base64

import cv2
import numpy as np


kernel = np.ones((10,10),np.float32)/100

def augment_frame(event, frame):
    if 'target_locked' in event and event['target_locked']:
        if 'target_box' in event:
            ## show frame, smoothing everything except target box
            # convert coordinates of the bounding box to pixels
            height, width, _ = frame.shape
            [y1, x1, y2, x2] = event['target_box']
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)

            # apply smoothing filter, saving the face
            face = frame[y1:y2, x1:x2]
            frame = cv2.filter2D(frame, -1, kernel)        
            frame[y1:y2, x1:x2] = face

            # draw blue rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 1)
        else:
            ## target box lost - show frame, smoothing everything
            frame = cv2.filter2D(frame, -1, kernel)        
    return frame
    
def run(fileh, delay):
    while True:
        data = fileh.readline()

        # extract event structure
        event = json.loads(data)
        # decode frame back into OpenCV object
        imdata = base64.b64decode(event['frame'])
        frame = cv2.imdecode(np.frombuffer(imdata, np.uint8), -1)

        frame = augment_frame(event, frame)

        # Put event attributes on the image
        def putText(row, text):
            scale = 0.5
            cv2.putText(frame, text, (0, int(25*row*scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, 255)
        putText(1, 'camera_moves=%s' % event['camera_moves'])
        if 'target_locked' in event:
            putText(2, 'target_locked=%s' % event['target_locked'])
        if 'target_box' in event:
            tb = list(map(lambda x: ("%.3f" % x), event['target_box']))
            putText(3, 'target_box=%s' % tb)

        # Display
        cv2.imshow('frame', frame)

        if delay is not None:
            if cv2.waitKey(delay) == 'q':
                break

def open_downstream_port(port):
    HOST = ''
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, port))
    s.listen(1)
    print('Socket now listening')
    conn, addr = s.accept()
    print('Connection accepted')
    return conn.makefile('rb')

def open_downstream_file(file):
    return open(file, 'rb')

if __name__ == "__main__":
    import sys

    try:
        assert(len(sys.argv) == 2)
        try:
            port = int(sys.argv[1])
            delay = 1 # minimal delay, show received frames as fast as possible
        except:
            port = None
            delay = 10 # FIXME fixed 10 ms delay (~100 fps)

    except:
        print("")
        print("Usage: %s (upstream-port|upstream-file)")
        print("    %s -1" % (sys.argv[0]))
        print("    %s 0 upstream.json" % (sys.argv[0]))
        exit(1)

    if port is not None:
        fileh = open_downstream_port(port)
    else:
        fileh = open_downstream_file(sys.argv[1])
    run(fileh, delay)
