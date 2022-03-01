import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def compute_learning_rate(lr, epoch):
    if callable(lr):
        return lr(epoch)
    else:
        return lr


def get_next_batch_supervised(batch_size, batch_idx, x, y):
    start_idx = batch_idx * batch_size

    if (start_idx > (len(x) - 1)):
        start_idx = start_idx % (len(x) - 1)

    end_idx = start_idx + batch_size

    if (end_idx > (len(x) - 1)):
        x_batch = x[start_idx:] + x[:(end_idx - len(x))]
        y_batch = y[start_idx:] + y[:(end_idx - len(x))]
    else:
        x_batch = x[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

    return x_batch, y_batch

def train_supervised(model, batch_size, epochs, x_train, y_train, x_val, y_val, lr=1e-4, keep_prob=1.0, save_path=None,
                     plot=False, sess_path=None):
    # Saver
    saver = tf.train.Saver()

    # Log variables
    train_acc_arr = []
    val_acc_arr = []
    train_loss_arr = []
    val_loss_arr = []

    min_val_acc = 0

    with tf.Session() as sess:
        if sess_path == None:
            # Start new session
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, sess_path)

        for epoch in range(epochs):
            x_train, y_train = shuffle(x_train, y_train)

            for batch_idx in range(len(x_train) // batch_size):
                x_batch, y_batch = get_next_batch_supervised(batch_size, batch_idx, x_train, y_train)
                if hasattr(model, 'keep_prob'):
                    model.training_op.run(feed_dict={model.x: x_batch,
                                                     model.y: y_batch,
                                                     model.learning_rate: compute_learning_rate(lr, epoch),
                                                     model.keep_prob: keep_prob})
                else:
                    model.training_op.run(feed_dict={model.x: x_batch,
                                                     model.y: y_batch,
                                                     model.learning_rate: compute_learning_rate(lr, epoch)})

            if hasattr(model, 'keep_prob'):
                train_loss, train_acc = sess.run([model.loss, model.accuracy],
                                                 feed_dict={model.x: x_train,
                                                            model.y: y_train,
                                                            model.keep_prob: 1.0})
                val_loss, val_acc, pred = sess.run([model.loss, model.accuracy, model.pred],
                                                   feed_dict={model.x: x_val,
                                                              model.y: y_val,
                                                              model.keep_prob: 1.0})
            else:
                train_loss, train_acc = sess.run([model.loss, model.accuracy],
                                                 feed_dict={model.x: x_train,
                                                            model.y: y_train})
                val_loss, val_acc, pred = sess.run([model.loss, model.accuracy, model.pred],
                                                   feed_dict={model.x: x_val,
                                                              model.y: y_val})

            if plot:
                train_loss_arr.append(train_loss)
                val_loss_arr.append(val_loss)

                train_acc_arr.append(train_acc)
                val_acc_arr.append(val_acc)

            if min_val_acc < val_acc:
                min_val_acc = val_acc
                if save_path != None:
                    saver.save(sess, save_path)

            print("epoch: ", epoch, "train acc:", train_acc, "train loss", train_loss,
                  "val acc", val_acc, "val_loss", val_loss, "best val acc", min_val_acc)

        if plot:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(range(epochs), train_acc_arr, label="train_acc")
            ax[0].plot(range(epochs), val_acc_arr, label="val_acc")
            ax[0].legend()
            ax[0].set_title("Accuracy")
            ax[0].set_ylim(0, 1)
            ax[0].set_xlim(0, epochs - 1)

            ax[1].plot(range(epochs), train_loss_arr, label="train_loss")
            ax[1].plot(range(epochs), val_loss_arr, label="val_loss")
            ax[1].legend()
            ax[1].set_title("Loss")
            ax[1].set_ylim(0, 0.1 + max(max(val_loss_arr), max(train_loss_arr)))
            ax[1].set_xlim(0, epochs - 1)
