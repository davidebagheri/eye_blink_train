# eye_segmentation_lstm

This is an implementation of an LSTM based classifier to detect bliks out
of a timeseries of eye statistics (segmentation, landmarks).

The code has been tested on Tensorflow 1.0.0 and Python 3.6

## Dataset

The dataset is built recording with a RGB camera some people blinking. For each frame the 
segmentation, landmarks and label (eye closed/open) have been annotated and stored in csv files.  

The dataset can be retrieved upon request at https://drive.google.com/drive/folders/1zdkGIJd4ibfnMLBAid7nVBOPB-99cj5a .

## Train

Install the requirements with

```
pip install -r requirements.txt
```

and train the LSTM using this command 

```
python train.py -d <path-to-dataset> -s <path-where-the-trained-model-will-be-saved> [ADDITIONAL BOOLEAN OPTIONS]

ADDITIONAL BOOLEAN OPTIONS:
  --use_ear : use Eye Aspect Ratio as input to the LSTM
  --use_seg_area : use Segmentation  as input to the LSTM
  --use_landmark_pos : to use the Landmarks position as input to the LSTM
```