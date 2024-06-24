# Draft plan: Cubix - The AI Rubik's Cube Detector Model

## Preparing the data

- collection of inputs, that needs to be labeled
  - lableme is used to label the images
- split the data into training and testing sets
- 90/10 split training/testing  (90% training, 10% testing)

## Training the model
- copy modelgarden to get the tensorflow Object Detection API 
- in the repo, there is a slection of pre-training model that we could take adventage of it instead of building it from scratch.
- We took our own inputs, and feed into it, start training the model
- We can use the Tensorboard to monitor the training process
 
Setting up 


## May 20 
### Tech stack
tensorflow
Javascript / python? 

### Model

The goal of this project is to train a model that can detect the colors of a Rubik's cube. The model will be trained on a dataset of images of Rubik's cubes. The model will be able to detect the colors of the Rubik's cube in real-time.

In order to do this we will train the model on a dataset of images of Rubik's cubes.

This project will explore which techniques are best for training a model to detect the colors of a Rubik's cube.


### How to photo Cubes for training 
- How many images should we need? 

- How to take the images?(background, lighting, angle, distance, etc.)

- What is the feature and label taht we provide ? 
- How to label the images?
- How to split the data into training and testing sets?
- How to augment the data?


- Each face has a center piece, and one center piece, four corner, and four edge pieces.
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb#scrollTo=he5u_okAYS4a

tf.keras API, the layers you mentioned—Flatten, Dense, and Dropout—eac

### Brainstorming

for CFOP training model: https://www.kaggle.com/datasets/antbob/rubiks-cube-cfop-solutions/data



they generated 10000 datasets to train this model 🤯 they did not do it by hand though.. looks like they made use of a python library called `PyCuber` which generated their datasets for them

- a bit of a cheat but I guess if it exists already why build all this manually lol
- I wonder if this can be broken down into individual models / steps some how instead if you want to make a "trainer"?
- a model per "State"? Cross, First two layers, Orientation of the last layer, Permutation of the last layer
- or perhaps it can be labeled to detect what state its in and continue from there
maybe we can feel in less training data if we use something like jperms as its training data? 🤷‍♂️
NOTE this is for solving the cube

We are doing a step they don't do, which is trying to actually detect and current cube state..

so I imagine there is at least two steps / models to train
- cube detection model
- CFOP solving model

#2 could also be solved via algos / library (such as the pycuber library) if we dont want to build that right away 
 

https://www.kaggle.com/datasets/antbob/rubiks-cube-cfop-solutions/data


## May 21 
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

1. Create a new Anaconda virtual environment

```
conda create -n tensorflow pip python=3.9
```


2. Activate the virtual environment

```
conda activate tensorflow
```

### TensorFlow Installation
```
pip install --ignore-installed --upgrade tensorflow
```


### Verify your Installation
Run the following command in a Terminal window:
```
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))
```

### Protobuf Installation/Compilation
```bash
Download the latest protoc-*-*.zip release (e.g. protoc-3.12.3-win64.zip for 64-bit Windows)

 => https://github.com/protocolbuffers/protobuf/releases => assets => protoc-27.0-rc-3-osx-aarch_64.zip


install this for architechture we are running the code 

```bash
Extract the contents of the downloaded protoc-*-*.zip in a directory <PATH_TO_PB> of your choice (e.g. C:\Program Files\Google Protobuf)
```
 
Add <PATH_T1O_PB>\bin to your Path environment variable (see Environment Setup)


```bash
go to .zprofile add the path there. 
export PATH="~/binary/protoc-27.0-rc-3-osx-aarch_64/bin:/usr/local/opt/python/libexec/bin:$PATH"



eval "$(/opt/homebrew/bin/brew shellenv)"
```
 
 
```
# install imagemagick
brew install imagemagick

# convert a single image
magick convert foo.HEIC foo.jpg

# bulk convert multiple images
magick mogrify -strip -monitor -format png *.JPG
```


python model_main_tf2.py --model_dir=models/custom_ssd_resnet50_v1_fpn --pipeline_config_path=models/custom_ssd_resnet50_v1_fpn/pipeline.config


tensorboard --logdir=models/custom_ssd_resnet50_v1_fpn

python model_main_tf2.py --model_dir=models/custom_ssd_resnet50_v1_fpn --pipeline_config_path=models/custom_ssd_resnet50_v1_fpn/pipeline.config --checkpoint_dir=models/custom_ssd_resnet50_v1_fpn

export model



we build a objecvct detection model by following this tutorial, https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html and 



May 24 Update 

also out of curiosity I reordered the labels putting the face last.. idk if that changed anything but I did get different results.. I might change it back to see if it will behave differently again...

the model for sure needs more input data
its only seeing Red right now, and it thinks white is red
interesting enough this is how that one github project I found does their labels too
https://github.com/Hemant-Mulchandani/Rubiks-Cube-Face-Detection-Model/blob/main/annotations/label_map.pbtxt
again no idea if the order matters at all (I think it doesnt) I just think we didnt give it enough training data for it to figure out everything

seems like it only found face before because it was first.. now its only finding red..

screenshots/Screenshot 2024-05-25 at 11.14.09 AM.png
![](screenshots/Screenshot 2024-05-25 at 11.14.09 AM.png)
![](screenshots/Screenshot 2024-05-25 at 11.14.12 AM.png)


May 29 


when running python image_object_detection_saved_model.py / image_object_dection_checkpoint.py / camera_object_detection.py:
you can update the parameters for the max boxes to draw and min score (so you dont see a lot of clutter! lots of 30% matches etc...)
          max_boxes_to_draw=200,
          min_score_thresh=.30,

could be something like this instead:
          max_boxes_to_draw=20,
          min_score_thresh=.90,

 

June

 ```
 python test_camera_detection.py 

 tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve ./exported-models/rubiks_model/saved_model ./tfjs
 ```
June 7-10

there seems to be an issue when exporting the model to tensorflowjs, the model is not working as expected. switching to other versions of tensorflow and retraining still appeared to give other problems.  attempting to switch to TF Vision also had some issues when trying to export (crashing).  In my testing with a sample model, switching to a docker container with preinstalled tensorflow and cuda components isntalled seemms to be more succesfull so far.  There was still some issues with tensorflowjs pypi library as it wanted to downgrade tensorflow when installing but creating a custom docker image to prevent this seems to work so far

I am able to succesfully export to a tensorflowjs model now, however I have not tested this yet with the rubiks model, I will need to migrate this code to work with TF-Vision instead

- Alternatively since there was more success with the docker contrainer, I could try downgrading to an older container to see if this current project will build successfully there


## Tensorflow js Models garden 
- research floder is under maintain .....
- switch to TF version 







labelme json format per file  => a single coco json format 
in order to finally using tf version tool to convert to a tfrecord file. 






