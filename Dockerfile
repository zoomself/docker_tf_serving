FROM tensorflow/serving
MAINTAINER zoomself.chn@gmail.com
EXPOSE 8501

ADD saved_model_sample models/sample
ADD saved_model_mnist models/mnist

ENV MODEL_NAME mnist
