from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.utils.object_detection import visualization_utils
from PIL import Image
from six import BytesIO

from shared import category_index, show_batch, tf_ex_decoder, export_dir, HEIGHT, WIDTH


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = None
    if (path.startswith('http')):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(image_data))

    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)


def build_inputs_for_object_detection(image, input_image_size):
    """Builds Object Detection model inputs for serving."""
    image, _ = resize_and_crop_image(
        image,
        input_image_size,
        padded_size=input_image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    return image


num_of_examples = 3

test_data_input_path = './tfrecords/test-00000-of-00001.tfrecord'

test_ds = tf.data.TFRecordDataset(
    test_data_input_path).take(
        num_of_examples)
show_batch(test_ds)

imported = tf.saved_model.load(export_dir)
model_fn = imported.signatures['serving_default']

input_image_size = (HEIGHT, WIDTH)
plt.figure(figsize=(20, 20))
# Change minimum score for threshold to see all bounding boxes confidences.
min_score_thresh = 0.30

for i, serialized_example in enumerate(test_ds):
    plt.subplot(1, 3, i+1)
    decoded_tensors = tf_ex_decoder.decode(serialized_example)
    image = build_inputs_for_object_detection(
        decoded_tensors['image'], input_image_size)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype=tf.uint8)
    image_np = image[0].numpy()
    result = model_fn(image)
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        result['detection_boxes'][0].numpy(),
        result['detection_classes'][0].numpy().astype(int),
        result['detection_scores'][0].numpy(),
        category_index=category_index,
        use_normalized_coordinates=False,
        max_boxes_to_draw=200,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=4)

    im = Image.fromarray(image_np)
    im.save(f'SerializedExampleImage-{i+1}.png')
