# eye_segmentation_unet

This sub-package contains an implementation of Unet for an eye segmentation application.

The code has been tested on Tensorflow 1.14.0 and Python 3.6

## Data

Dataset from https://arxiv.org/pdf/1910.05283.pdf has been used. It contains more than 8000 eye images with
corresponding segmentation (eye, sclera and pupil+iris)

![Alt text](images/image.png?raw=true "Title") ![Alt text](images/ground_truth.png?raw=true "Title")


## Train

Install the requirements with

```
pip install -r requirements.txt
```

and train the Unet using this command 

```
python train.py -d <path-to-dataset> -s <path-where-the-trained-model-will-be-saved>
```

a figure with training statistics (accuracy, loss), will be additionally saved

![Alt text](images/plot.png?raw=true "Title")