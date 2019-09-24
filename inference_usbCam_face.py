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
        #start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        #print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)

def usage():
    print("")
    print("Usage: %s cameraID (upstream-port|upstream-file)")
    print("    %s 0 0" % (sys.argv[0]))
    print("    %s 0 8089" % (sys.argv[0]))
    print("    %s 0 upstream.json" % (sys.argv[0]))
    exit(1)

import profiler
profiler = profiler.Profiler()

pb = planb.PlanB()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--stream_frames", help="do send encoded frames upstream")
    parser.add_argument("camera", help="do not send frames upstream")
    parser.add_argument("upstream", help="do not send frames upstream")
    args = parser.parse_args()

    if not args.camera.isdigit():
        usage()
    camID = int(args.camera)

    if args.upstream.isdigit():
        port = int(args.upstream)
        if port > 0:
            pb.upstream.connect_socket('localhost', port)
        # else port == 0: do nothing
    else:
        pb.upstream.open_file(args.upstream)
    
    if args.stream_frames:
        pb.upstream.stream_frames(args.stream_frames)

    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    cap = cv2.VideoCapture(camID)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    windowName = None
    visualize = True

    print("Visualize = %s" % visualize)

    while True:
        t = [time.time(), 0, 0, 0, 0, 0]

        ret, image = cap.read()
        if ret == 0:
            break
        t[1] = time.time()

        meta = {}

        #res = pb.process_image(image, meta)
        image = cv2.flip(image, 1)

        t[2] = time.time()

        if pb.is_camera_moving():
            pb.upstream.send_event_camera_moving(image, meta)
            pb.release_target()
        else:
            (boxes, scores, classes, num_detections) = tDetector.run(image)
            t[3] = time.time()

            pb.process_boxes(image, meta, boxes[0], scores[0])
            t[4] = time.time()

            if visualize:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4)

        if visualize:
            if windowName is None:
                [h, w] = image.shape[:2]
                windowName = "tensorflow based (%d, %d)" % (w, h)
                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            pb.show_info(image)
            cv2.imshow(windowName, image)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            print("Exiting...")
            break
        elif k == ord('r'):
            print("Resetting move timing (%d/%d)" % (pb.move_duration, pb.move_period))
            pb.reset_move()
        elif k == ord('v'):
            visualize = not visualize
            print("Visualize = %s" % visualize)

        t[5] = time.time()
        profiler.report(t)

    profiler.close()
    pb.upstream.close()
    cap.release()
