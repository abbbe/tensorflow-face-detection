#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

import planb

from utils import label_map_util
from utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True


    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        #print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)

import profiler
profiler = profiler.Profiler()

if __name__ == "__main__":
    import sys

    try:
        assert(len(sys.argv) == 3)
        camID = int(sys.argv[1])

        try:
            upstreamPort = int(sys.argv[2])
            upstreamFile = None
        except:
            upstreamPort = -1
            upstreamFile = sys.argv[2]

    except:
        print("")
        print("Usage: %s cameraID (upstream-port|upstream-file)")
        print("    %s 0 -1" % (sys.argv[0]))
        print("    %s 0 8089" % (sys.argv[0]))
        print("    %s 0 upstream.json" % (sys.argv[0]))
        exit(1)

    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    planb.connect_upstream(port=upstreamPort, file=upstreamFile)

    cap = cv2.VideoCapture(camID)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    windowNotSet = True
    while True:
        t = [time.time(), 0, 0, 0, 0, 0]

        ret, image = cap.read()
        if ret == 0:
            break
        t[1] = time.time()

        [h, w] = image.shape[:2]

        meta = {}

        res = planb.process_image(image, meta)
        t[2] = time.time()

        if res:
            image = cv2.flip(image, 1)

            (boxes, scores, classes, num_detections) = tDetector.run(image)
            t[3] = time.time()

            planb.process_boxes(image, meta, boxes[0], scores[0])
            t[4] = time.time()

            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4)

            if windowNotSet is True:
                cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
                windowNotSet = False

            cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
            k = cv2.waitKey(1) & 0xff
            if k == ord('q') or k == 27:
                break
                
        t[5] = time.time()
        profiler.report(t)

        if cv2.waitKey(1) == 'q': # FIXME
            break

    profiler.close()
    planb.deinit()
    cap.release()
