### Dockerfile
 
This file is used to create a reproducing environment for the project. To train the model on a GPU

#### Buiilding and running
To build the docker image run this command
```bash
sudo docker build --tag 'tf_vision' .
```

to use it, we can create a shell into the docker image to the scripts to train and visualize the model
```bash
sudo docker run  -it --rm --runtime=nvidia --gpus all -v $PWD:/app -w /app 'tf_vision' bash
python trainer.py
```

### Reference

This reference was used in order to create this project and adapt for rubix cubes

https://www.tensorflow.org/tfmodels/vision/object_detection

### Converting images
all images should be converted to strip out any unwanted metadata

```bash
# install imagemagick
brew install imagemagick

# bulk convert multiple images striping meta data
magick mogrify -strip -monitor -format jpg *.JPG
```

### convert labelme json files to coco json files

```bash 
python labelme2coco.py
```

### CLI command to convert data(train data).
```bash
ROOT_FOLDER="./images"
TRAIN_DATA_DIR="${ROOT_FOLDER}/train"
TRAIN_ANNOTATION_FILE_DIR="${TRAIN_DATA_DIR}/_annotations.coco.json"
OUTPUT_TFRECORD_TRAIN="./tfrecords/train"

# Need to provide
  # 1. image_dir: where images are present
  # 2. object_annotations_file: where annotations are listed in json format
  # 3. output_file_prefix: where to write output convered TFRecords files
python -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir=${TRAIN_DATA_DIR} \
  --object_annotations_file=${TRAIN_ANNOTATION_FILE_DIR} \
  --output_file_prefix=$OUTPUT_TFRECORD_TRAIN \
  --num_shards=1
  ```
### CLI command to convert data(validation data).
```bash
VALID_DATA_DIR="${ROOT_FOLDER}/valid"
VALID_ANNOTATION_FILE_DIR="${VALID_DATA_DIR}/_annotations.coco.json"
OUTPUT_TFRECORD_VALID="./tfrecords/valid"

python -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir=$VALID_DATA_DIR \
  --object_annotations_file=$VALID_ANNOTATION_FILE_DIR \
  --output_file_prefix=$OUTPUT_TFRECORD_VALID \
  --num_shards=1
```
### CLI command to convert data(test data).
```bash
TEST_DATA_DIR="${ROOT_FOLDER}/test"
TEST_ANNOTATION_FILE_DIR="${TEST_DATA_DIR}/_annotations.coco.json"
OUTPUT_TFRECORD_TEST='./tfrecords/test'

python -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir=$TEST_DATA_DIR \
  --object_annotations_file=$TEST_ANNOTATION_FILE_DIR \
  --output_file_prefix=$OUTPUT_TFRECORD_TEST \
  --num_shards=1
```


### Web model conversion

tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve ./exported_model ./web_model
