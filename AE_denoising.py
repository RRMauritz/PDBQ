import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras import optimizers
from Backend import jsd, PrinterCallback
from DataConversion import noise


def run_ae_denoising(testGT, train_corr, test_corr, structure, v, n, e, bs):
    """"
    Implementation of a denoising autoencoder (DAE).
    :param testGT: the ground truth test data (no corruptions)
    :param train_corr: the corrupted train data
    :param test_corr: the corrupted test data
    :param structure: the structure of the train and test data
    :param v: the fraction of attributes that is corrupted in the corruption process of the DAE
    :param n: the size of the hidden layer in the DAE
    :param e: the number of epochs used for training the DAE
    :param bs: the batch sizes used for training the DAE

    :returns the outputs from the DAE based on the corrupted test-data and the history of the trained DAE
    """
    m = len(structure)
    structure_0 = [0] + structure

    # Convert the data to a format so that in can be used as target in the autoencoder
    train_attr_corr = [train_corr[:, i:j] for i, j in zip(np.cumsum(structure_0[:-1]), np.cumsum(structure_0[1:]))]
    test_attr_GT = [testGT[:, i:j] for i, j in zip(np.cumsum(structure_0[:-1]), np.cumsum(structure_0[1:]))]

    # Corrupt the train_corr data and use the 'clean' version as target
    # NOTE: noisy is not the same as train_corr! noisy can be seen as double corrupted (corrupted version of train_corr)
    noisy = noise(train_corr, structure, v)  # v% of the attribute values is set to 'missing value'

    # Autoencoder 1 construction and training ----------------------------------------------------------------------
    inputs = Input(shape=(train_corr.shape[1],))
    encoded = Dense(n, activation='relu')(inputs)
    decodes = [Dense(q, activation='softmax')(encoded) for q in structure]
    losses = [jsd for o in range(m)]  # JSD loss function for each attribute (having a number of categories sum = 1)
    autoencoder = Model(inputs, decodes)
    sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    autoencoder.compile(optimizer=sgd, loss=losses, loss_weights=[1 for k in range(m)])

    # history = autoencoder.fit(noisy, train_attr_corr, epochs=e, batch_size=bs, shuffle=True, verbose=0,
    #                            validation_data=(test_corr, test_attr_GT), callbacks=[PrinterCallback()])
    history = autoencoder.fit(noisy, train_attr_corr, epochs=e, batch_size=bs, shuffle=True, verbose=0)

    # TODO: IMPORTANT: using the test-set for validation is wrong as we are not allowed to take decision on it!

    # Output the results -------------------------------------------------------------------------------------------
    predictions = autoencoder.predict(test_corr)
    outputs = np.concatenate([np.round(predictions[i], 2) for i in range(m)], axis=1)
    return outputs, history
