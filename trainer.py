import pprint

import tensorflow as tf
import tensorflow_models as tfm
from official.core import exp_factory
from official.vision.serving import export_saved_model_lib

from shared import show_batch, export_dir, HEIGHT, WIDTH

pp = pprint.PrettyPrinter(indent=4)  # Set Pretty Print Indentation
print(tf.__version__)  # Check the version of tensorflow used


train_data_input_path = './tfrecords/train-00000-of-00001.tfrecord'
valid_data_input_path = './tfrecords/valid-00000-of-00001.tfrecord'
model_dir = './trained_model/'

exp_config = exp_factory.get_exp_config('retinanet_mobile_coco')

batch_size = 1
num_classes = 7

IMG_SIZE = [HEIGHT, WIDTH, 3]

# Backbone config.
exp_config.task.freeze_backbone = False
exp_config.task.annotation_file = ''

# Model config.
exp_config.task.model.input_size = IMG_SIZE
exp_config.task.model.num_classes = num_classes + 1
exp_config.task.model.detection_generator.tflite_post_processing.max_classes_per_detection = exp_config.task.model.num_classes

# Training data config.
exp_config.task.train_data.input_path = train_data_input_path
exp_config.task.train_data.dtype = 'float32'
exp_config.task.train_data.global_batch_size = batch_size
exp_config.task.train_data.parser.aug_scale_max = 1.0
exp_config.task.train_data.parser.aug_scale_min = 1.0

# Validation data config.
exp_config.task.validation_data.input_path = valid_data_input_path
exp_config.task.validation_data.dtype = 'float32'
exp_config.task.validation_data.global_batch_size = batch_size


logical_device_names = [
    logical_device.name for logical_device in tf.config.list_logical_devices()]

if 'GPU' in ''.join(logical_device_names):
    print('This may be broken in Colab.')
    device = 'GPU'
elif 'TPU' in ''.join(logical_device_names):
    print('This may be broken in Colab.')
    device = 'TPU'
else:
    print('Running on CPU is slow, so only train for a few steps.')
    device = 'CPU'


train_steps = 10000
# steps_per_loop = num_of_training_examples // train_batch_size
exp_config.trainer.steps_per_loop = 100

exp_config.trainer.summary_interval = 100
exp_config.trainer.checkpoint_interval = 100
exp_config.trainer.validation_interval = 100
# validation_steps = num_of_validation_examples // eval_batch_size
exp_config.trainer.validation_steps = 100
exp_config.trainer.train_steps = train_steps
exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 100
exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05


if exp_config.runtime.mixed_precision_dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

if 'GPU' in ''.join(logical_device_names):
    distribution_strategy = tf.distribute.MirroredStrategy()
elif 'TPU' in ''.join(logical_device_names):
    tf.tpu.experimental.initialize_tpu_system()
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu='/device:TPU_SYSTEM:0')
    distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    print('Warning: this will be really slow.')
    distribution_strategy = tf.distribute.OneDeviceStrategy(
        logical_device_names[0])

print('Done')


with distribution_strategy.scope():
    task = tfm.core.task_factory.get_task(
        exp_config.task, logging_dir=model_dir)


for images, labels in task.build_inputs(exp_config.task.train_data).take(1):
    print()
    print(
        f'images.shape: {str(images.shape):16}  images.dtype: {images.dtype!r}')
    print(f'labels.keys: {labels.keys()}')


model, eval_logs = tfm.core.train_lib.run_experiment(
    distribution_strategy=distribution_strategy,
    task=task,
    mode='train_and_eval',
    params=exp_config,
    model_dir=model_dir,
    run_post_eval=True)

export_saved_model_lib.export_inference_graph(
    input_type='image_tensor',
    batch_size=1,
    input_image_size=[HEIGHT, WIDTH],
    params=exp_config,
    checkpoint_path=tf.train.latest_checkpoint(model_dir),
    export_dir=export_dir)

buffer_size = 20
num_of_examples = 3

raw_records = tf.data.TFRecordDataset(
    exp_config.task.train_data.input_path).shuffle(
        buffer_size=buffer_size).take(num_of_examples)
show_batch(raw_records)
