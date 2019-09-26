import socket
import struct
import json
import base64
import cv2

import logging
logger = logging.getLogger('upstream')
logger.setLevel(logging.INFO)

class Upstream:
    T1 = 3
    T2 = 10
    T3 = 3

    def __init__(self):
        self.stream_frames = False
        self.clientsocket = None
        self.fileh = None

    def connect_socket(self, host, port, stream_frames):
        self.stream_frames = stream_frames
        self.clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientsocket.connect((host, port))
        logger.info('Connected to upstream server %s:%d' % (host, port))

    def open_file(self, file, stream_frames):
        self.stream_frames = stream_frames
        self.fileh = open(file, 'wb')
        logger.info('Opened upstream file %s' % file)

    def send_idle(self, image, meta):
        self.send_event(image, meta, { 'mode': 'idle' })

    def send_effect_start(self, image, meta, box):
        self.send_event(image, meta, { 'mode': 'effect_start', 'box': box.tolist(), 'time': [self.T1, self.T2, self.T3] })

    def send_effect_run(self, image, meta, box):
        self.send_event(image, meta, { 'mode': 'effect_run', 'box': box.tolist() })

    def send_effect_abort(self, image, meta):
        self.send_event(image, meta, { 'mode': 'effect_abort', 'time': [self.T3] })

    def send_event(self, image, meta, event):
        logger.debug('send_event(%s)' % event)

        if self.clientsocket is not None or self.fileh is not None:
            if self.stream_frames:
                # add encoded frame to the event structure
                image = cv2.resize(image, (640, 480)) # FIXME
                _, imdata = cv2.imencode('.jpg', image)
                encoded_imdata = base64.b64encode(imdata).decode('ascii')
                event['frame'] = encoded_imdata

            # if self.stream_timestamps:
            #   events['t'] = ...

            msg = (json.dumps(event) + '\r\n').encode('ascii')

            # send it upstream
            if self.clientsocket is not None:
                self.clientsocket.sendall(msg)
            if self.fileh is not None:
                self.fileh.write(msg)
                self.fileh.flush()

    def close(self):
        if self.clientsocket is not None:
            self.clientsocket.close()
            self.clientsocket = None
        if self.fileh is not None:
            self.fileh.close()
            self.fileh = None
