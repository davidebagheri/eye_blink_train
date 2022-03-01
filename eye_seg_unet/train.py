import tensorflow as tf
from unet import unet
from eye_seg_data_loader import EyeSegDataLoader
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", default=None, help="Data path to Eye_Segmentation_Database")
    parser.add_argument("-s", "--saved_model_path", default=None, help="Path in which the trained model is saved")
    args = parser.parse_args()

    if args.data_path == None:
        print("Give the path to the dataset!")
        exit()

    # Load model
    model = unet(3, input_shape=(40,80,3))

    # Load data
    data_loader = EyeSegDataLoader(args.data_path)

    # Reshape gt image for training
    y_train_shape = data_loader.y_train.shape
    y_val_shape = data_loader.y_test.shape

    data_loader.y_train = data_loader.y_train.reshape((y_train_shape[0], y_train_shape[1]*y_train_shape[2], y_train_shape[3]))
    data_loader.y_test = data_loader.y_test.reshape((y_val_shape[0], y_val_shape[1]*y_val_shape[2], y_val_shape[3]))

    # Callbacks
    callbacks = []

    # Learning rate callback
    def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 15:
            lr *= 1e-1
        elif epoch > 40:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-3

        print('Learning rate: ', lr)
        return lr


    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    callbacks.append(lr_callback)

    # Save Callback
    if args.saved_model_path != None:
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="seg_unet",
            monitor='val_acc',
            mode='max',
            save_weights_only=True,
            save_best_only=True)
        callbacks.append(save_callback)

    # Train
    epochs = 100
    batch_size = 32

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    history = model.fit(x=data_loader.x_train,
                        y=data_loader.y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(data_loader.x_test,
                                         data_loader.y_test)
                        )

    # Plot Results
    acc = history.history["acc"]
    loss= history.history["loss"]
    val_acc = history.history["val_acc"]
    val_loss= history.history["val_loss"]

    plt.title('LOSS')
    plt.plot(loss,'-r', label='Train loss')
    plt.plot(val_loss,'-b', label='Validation loss')
    plt.legend()
    plt.ylim([0,1])
    plt.savefig("training_loss.png")

    plt.figure()
    plt.title('ACCURACY')
    plt.plot(acc,'-r', label='Train accuracy')
    plt.plot(val_acc,'-b', label='Validation accuracy')
    plt.ylim([0,1])
    plt.legend()
    plt.savefig("training_accuracy.png")
