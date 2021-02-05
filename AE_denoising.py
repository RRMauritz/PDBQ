import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras import optimizers
from Backend import jsd, PrinterCallback
from DataConversion import noise
from sklearn.model_selection import train_test_split


def run_ae_denoising(train, test, structure, v, n, e, bs):
    """"
    Implementation of a denoising autoencoder (DAE).
    :param train: the CORRUPTED train data
    :param test: the CORRUPTED test data
    :param structure: the structure of the train and test data
    :param v: the fraction of attributes that is corrupted in the corruption process of the DAE
    :param n: the size of the hidden layer in the DAE
    :param e: the number of epochs used for training the DAE
    :param bs: the batch sizes used for training the DAE

    :returns the outputs from the DAE based on the corrupted test-data and the history of the trained DAE
    """
    m = len(structure)
    structure_0 = [0] + structure

    # Split the train data in a train and validation set
    train_corr, val_corr = train_test_split(train, train_size=0.80)
    # Convert the data to a format so that in can be used as target in the autoencoder
    # train_attr_corr[i,j] = the one-hot encoding for category i of data-point j
    train_attr_corr = [train_corr[:, i:j] for i, j in zip(np.cumsum(structure_0[:-1]), np.cumsum(structure_0[1:]))]
    val_attr_corr = [val_corr[:, i:j] for i, j in zip(np.cumsum(structure_0[:-1]), np.cumsum(structure_0[1:]))]

    # Corrupt the train_corr data and use the 'clean' version as target
    # NOTE: this noise process is not the same as the corruption process resembling the uncertain in prob. data!
    # This is merely for the training process of the DAE
    train_corr_noisy = noise(train_corr, structure, v)
    val_corr_noisy = noise(val_corr, structure, v)

    # Autoencoder construction and training ----------------------------------------------------------------------
    inputs = Input(shape=(train_corr.shape[1],))
    encoded = Dense(n, activation='relu')(inputs)
    decodes = [Dense(q, activation='softmax')(encoded) for q in structure]
    losses = [jsd for o in range(m)]  # JSD loss function for each attribute (having a number of categories sum = 1)
    autoencoder = Model(inputs, decodes)
    sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    autoencoder.compile(optimizer=sgd, loss=losses, loss_weights=[1 for k in range(m)])

    history = autoencoder.fit(train_corr_noisy, train_attr_corr, epochs=e, batch_size=bs, shuffle=True, verbose=0,
                              validation_data=(val_corr_noisy, val_attr_corr), callbacks=[PrinterCallback()])

    # Output the results -------------------------------------------------------------------------------------------
    predictions = autoencoder.predict(test)
    outputs = np.concatenate([np.round(predictions[i], 2) for i in range(m)], axis=1)
    return outputs, history
