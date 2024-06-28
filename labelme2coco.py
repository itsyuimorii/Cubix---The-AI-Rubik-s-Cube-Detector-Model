import os
import json
import glob

from shared import category_index

def main():

    input_folder = os.path.join(".", "images")
    train_folder = os.path.join(input_folder, "train")
    test_folder = os.path.join(input_folder, "test")
    valid_folder = os.path.join(input_folder, "valid")

    output_file_name = "_annotations.coco.json"

    labelme_to_coco(train_folder, os.path.join(train_folder, output_file_name))
    labelme_to_coco(test_folder, os.path.join(test_folder, output_file_name))
    labelme_to_coco(valid_folder, os.path.join(valid_folder, output_file_name))

def labelme_to_coco(input_folder, output_file):
    """
    Convert a folder of LabelMe JSON files to COCO JSON format using a predefined category index.

    Args:
    - input_folder (str): Path to the folder containing LabelMe JSON files.
    - output_file (str): Output file name for the COCO JSON output.
    """
    images = []
    annotations = []

    image_id = 1
    annotation_id = 1

    # Process each JSON file in the input folder
    files = glob.glob(os.path.join(input_folder, "Cube*.json"))
    for json_file in files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Create COCO image entry
        image_info = {
            'id': image_id,
            'file_name': data['imagePath'],
            'height': data['imageHeight'],
            'width': data['imageWidth']
        }
        images.append(image_info)

        # Process annotations
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']

            # Find category_id based on label using category_index
            category_id = None
            for cat_id, cat_info in category_index.items():
                if label == cat_info['name']:
                    category_id = cat_id
                    break

            if category_id is None:
                continue  # Skip if label not found in category_index

            # Calculate bounding box coordinates
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            xmin = min(x_values)
            xmax = max(x_values)
            ymin = min(y_values)
            ymax = max(y_values)
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            # Create COCO annotation entry
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': [],
                'area': bbox_width * bbox_height,
                'bbox': [xmin, ymin, bbox_width, bbox_height],
                'iscrowd': 0  # 0 for regular annotations
            }
            annotations.append(annotation)
            annotation_id += 1

        image_id += 1

    # Create COCO JSON structure
    coco_output = {
        'info': {},
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': list(category_index.values())  # Use values from category_index
    }

    # Save to output JSON file
    with open(output_file, 'w') as outfile:
        json.dump(coco_output, outfile)

    print(f"Conversion completed. COCO JSON saved to {output_file}")

if __name__ == "__main__":
    main()
