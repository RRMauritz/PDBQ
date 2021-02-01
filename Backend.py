import keras.backend as K
import tensorflow as tf


# NOTE: Keras uses base e for the logarithms!

def jsd(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    m = (1 / 2) * (y_true + y_pred)

    return (1 / 2) * K.sum(y_true * K.log(y_true / m), axis=-1) + (1 / 2) * K.sum(y_pred * K.log(y_pred / m), axis=-1)


def prob_output(x):
    return K.abs(x) / K.sum(K.abs(x))


class PrinterCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print('EPOCH: {}, Train Loss: {:05.4f}, Val Loss: {:05.4f}'.format(epoch,
                                                               logs['loss'],
                                                               logs['val_loss']))

