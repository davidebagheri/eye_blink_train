from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


def unet(n_classes, input_shape = (256,256,3)):
    inputs = Input(input_shape)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop4))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv9)
    conv9 = Conv2D(n_classes, 3, activation = 'sigmoid', padding = 'same')(conv9)

    conv9_shape = conv9.shape.as_list()

    reshape = Reshape((conv9_shape[1]*conv9_shape[2], conv9_shape[3]))(conv9)

    model = Model(inputs=inputs, outputs = reshape)

    return model
