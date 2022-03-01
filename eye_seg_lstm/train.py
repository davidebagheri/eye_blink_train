import numpy as np
from sklearn.utils import shuffle
import argparse
from utils import train_supervised
from data import CSVDataReader
from model import LSTM_classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", required=True)
    parser.add_argument("-s", "--save_path", default=".")
    parser.add_argument("-e", "--use_ear", default=False)
    parser.add_argument("-s", "--use_seg_area", default=True)
    parser.add_argument("-l", "use_landmark_pos", default=False)

    args = parser.parse_args()

    """
    Data organization:
    0) ear
    1) pupil
    2) white
    3) backgroud
    4) landmark_0_x
    5) landmark_0_y
    6) landmark_1_x
    7) landmark_1_y
    8) landmark_2_x
    9) landmark_2_y
    10) landmark_3_x
    11) landmark_3_y
    12) landmark_4_x
    13) landmark_4_y
    14) landmark_5_x
    15) landmark_5_y
    """

    # Read and prepare data
    seq_len = 10
    data_reader = CSVDataReader()
    X, Y = data_reader.get_dataset_from_csv(args.dataset_path, seq_len,
                                            shuffle_data=False)

    X = X.astype(np.float32)
    Y = np.expand_dims(Y.astype(np.int64), 1)

    input_length = 0
    mask = [False] * X.shape[1]

    if args.use_ear:
        mask[0] = True
        input_length += 1
    if args.use_seg_area:
        mask[3] = True
        input_length += 1
    if args.use_landmark_pos:
        for i in range(4, len(mask)):
            mask[i] = True
            input_length += 1

    X = X[:, mask, :]

    # Create model
    model = LSTM_classifier(seq_len, input_length)

    # Shuffle and split train and validation sets
    X,Y = shuffle(X,Y)
    X = X.reshape(len(X), seq_len, input_length)
    train_pct = 0.8
    n_train = int(0.8 * len(Y))
    x_train = X[:n_train]
    y_train = Y[:n_train]
    x_val = X[n_train:]
    y_val = Y[n_train:]

    #-------------- Train ----------------#
    def get_learning_rate(epoch):
        if epoch<20:
            lr=0.0001
        elif epoch<1000:
            lr=0.0001
        elif epoch<5000:
            lr=0.000001
        elif epoch<30000:
          lr=0.0000001
        else:
          lr=0.00000001
        return lr


    n_epochs = 200
    batch_size = 16
    model_save_path = args.save_path

    train_supervised(model,
                     batch_size,
                     n_epochs,
                     x_train,
                     y_train,
                     x_val,
                     y_val,
                     lr=get_learning_rate,
                     save_path=None,
                     plot=True)
