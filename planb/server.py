import pickle
import socket
import struct

import cv2
import numpy as np

HOST = ''
PORT = 8089

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn, addr = s.accept()
print('Connection accepted')

data = b'' ### CHANGED
payload_size = struct.calcsize("L") ### CHANGED

kernel = np.ones((10,10),np.float32)/100

while True:

    # Retrieve message size
    while len(data) < payload_size:
        chunk = conn.recv(4096)
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        data += chunk

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

    # Retrieve all data based on message size
    while len(data) < msg_size:
        chunk = conn.recv(4096)
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        data += chunk

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Extract the message
    [meta, event, frame] = pickle.loads(frame_data)
    #event['camera_moves']
    if event['target_locked']:
        if 'target_box' in event:
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 1)
        else:
            # apply smoothing filter all over
            frame = cv2.filter2D(frame, -1, kernel)        
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
