# eye_blink_train

This repository contains two separated modules to train two Neural Networks:

* Unet (https://arxiv.org/abs/1505.04597) for eye segmentation
* LSTM for blink classification out of the segmentation timeseries

Both are developed using Tensorflow and contains instructions to retrieve the corresponding
dataset and train.

## Some results

Here are some examples showing the output segmentation of the trained Unet given the input image

![Alt text](images/trained_unet.png?raw=true "Title")