FROM tensorflow/tensorflow:2.16.1-gpu-jupyter
RUN pip install tf-models-official
RUN pip install tensorflowjs==4.20.0 tensorflow==2.16.1
