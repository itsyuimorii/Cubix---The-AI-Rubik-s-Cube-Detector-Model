# Draft plan: Cubix - The AI Rubik's Cube Detector Model

 
## May 20 
a
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