#!/usr/bin/env python
# coding: utf-8
"""
Real-time Object Detection From TF2 Saved Model
===============================================
"""

import os
import cv2
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

MODEL_NAME = 'rubiks_model'
PATH_TO_MODEL_DIR = os.path.join("exported-models", MODEL_NAME)
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

LABEL_FILENAME = 'label_map.pbtxt'
PATH_TO_LABELS = os.path.join("annotations", LABEL_FILENAME)

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Load the model
print('Loading model...', end='')
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')

# Load label map data
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Setup video
cap = cv2.VideoCapture(0)  # Use 0 for web cam
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

try:
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame from BGR to RGB for detection
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Detection
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # Visualization of the results of a detection.
        viz_utils.visualize_boxes_and_labels_on_image_array(
            frame,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.80,
            agnostic_mode=False)

        # Display output in BGR format as captured
        cv2.imshow('Object Detection', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
