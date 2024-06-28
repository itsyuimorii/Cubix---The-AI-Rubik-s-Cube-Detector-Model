import matplotlib.pyplot as plt
import numpy as np
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
from official.vision.utils.object_detection import visualization_utils
from PIL import Image

HEIGHT, WIDTH = 640, 640

export_dir = './exported_model/'

category_index = {
    1: {
        'id': 1,
        'name': 'face'
    },
    2: {
        'id': 2,
        'name': 'red_tile'
    },
    3: {
        'id': 3,
        'name': 'white_tile'
    },
    4: {
        'id': 4,
        'name': 'blue_tile'
    },
    5: {
        'id': 5,
        'name': 'orange_tile'
    },
    6: {
        'id': 6,
        'name': 'green_tile'
    },
    7: {
        'id': 7,
        'name': 'yellow_tile'
    }
}
tf_ex_decoder = TfExampleDecoder()


def show_batch(raw_records):
    plt.figure(figsize=(20, 20))
    use_normalized_coordinates = True
    min_score_thresh = 0.30
    for i, serialized_example in enumerate(raw_records):
        plt.subplot(1, 3, i + 1)
        decoded_tensors = tf_ex_decoder.decode(serialized_example)
        image = decoded_tensors['image'].numpy().astype('uint8')
        scores = np.ones(shape=(len(decoded_tensors['groundtruth_boxes'])))
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image,
            decoded_tensors['groundtruth_boxes'].numpy(),
            decoded_tensors['groundtruth_classes'].numpy().astype('int'),
            scores,
            category_index=category_index,
            use_normalized_coordinates=use_normalized_coordinates,
            max_boxes_to_draw=200,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False,
            instance_masks=None,
            line_thickness=4)
        im = Image.fromarray(image)
        im.save(f'ShowBatchImage-{i+1}.png')
