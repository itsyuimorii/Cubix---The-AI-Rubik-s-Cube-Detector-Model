import os
import json
import tensorflow as tf
import numpy as np
from PIL import Image
from labelme import utils

def create_tf_example(image_path, annotations, label_map):
    # Load image
    image = Image.open(image_path)
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    
    # Read image data
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()
    
    filename = os.path.basename(image_path).encode('utf8')
    image_format = b'png'
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # Parse annotations
    for shape in annotations['shapes']:
        points = shape['points']
        (xmin, ymin), (xmax, ymax) = points[0], points[1]
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        label = shape['label']
        classes_text.append(label.encode('utf8'))
        classes.append(label_map[label])
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example

def convert_labelme_folder_to_tfrecord(labelme_folder_path, output_path, label_map):
    writer = tf.io.TFRecordWriter(output_path)
    
    for filename in os.listdir(labelme_folder_path):
        if filename.endswith('.json'):
            json_path = os.path.join(labelme_folder_path, filename)
            with open(json_path) as json_file:
                data = json.load(json_file)
            image_path = os.path.join(labelme_folder_path, data['imagePath'])
            annotations = data

            tf_example = create_tf_example(image_path, annotations, label_map)
            writer.write(tf_example.SerializeToString())
    
    writer.close()

# Example usage:
train_folder_path = 'workspace/training_rubik_cube_detection/images/train'
test_folder_path = 'workspace/training_rubik_cube_detection/images/test'
train_tfrecord_path = 'workspace/training_rubik_cube_detection/annotations/train.tfrecord'
test_tfrecord_path = 'workspace/training_rubik_cube_detection/annotations/test.tfrecord'

# Define your label map
label_map = {
    'face': 1,
    'red_tile': 2,
    'white_tile': 3,
    'blue_tile': 4,
    'orange_tile': 5,
    'green_tile': 6,
    'yellow_tile': 7,
}

# Convert training data
convert_labelme_folder_to_tfrecord(train_folder_path, train_tfrecord_path, label_map)

# Convert testing data
convert_labelme_folder_to_tfrecord(test_folder_path, test_tfrecord_path, label_map)

# Create label_map.pbtxt
with open('workspace/training_rubik_cube_detection/annotations/label_map.pbtxt', 'w') as f:
    for label, id in label_map.items():
        f.write('item {\n')
        f.write('  id: {}\n'.format(id))
        f.write('  name: \'{}\'\n'.format(label))
        f.write('}\n')
