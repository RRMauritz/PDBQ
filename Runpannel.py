import numpy as np
import pandas as pd
import os
from DataQuality import Q_JSD, Q_H, Q_faul
from AE_denoising import run_ae_denoising
from BN_sup import run_bn_sup
from DataConversion import dataConversion


def results(outputs, testGT, test_corr, corr_ind_test, model: str):
    print('   Summary for ', model, ':\n')
    print('   AVG JSD = ', np.round(Q_JSD(testGT, outputs), 5))
    print('   AVG JSD ocr = ', np.round(Q_JSD(testGT[corr_ind_test, :], outputs[corr_ind_test, :]), 5))
    a, avg_a = Q_faul(testGT, test_corr, outputs)
    b, avg_b = Q_faul(testGT[corr_ind_test, :], test_corr[corr_ind_test, :], outputs[corr_ind_test, :])
    print('   Entropy = ', Q_H(outputs))
    print('   Number of fauls = ', a, ' = ', np.round((a / test_corr.shape[0]) * 100, 2), '% average faul = ',
          np.round(avg_a, 5))
    print('   Number of fauls ocr = ', b, ' = ', np.round((b / len(corr_ind_test)) * 100, 2), '% average faul = ',
          np.round(avg_b, 5), '\n\n')


# ['2', '3', '4', '6', '9']

for version in ['9']:
    print('Model ' + version + '--------------------------------------------')
    dir = r"C:\Users\Rutger Mauritz\Google Drive\Studie Toegepaste Wiskunde\BSc\Module 12\Bachelor Assignment\Datasets" + '\Model' + version
    data = pd.read_csv(os.path.join(dir, 'Data' + version + '.csv')).values

    GT, data_corr, structure, corr_ind = dataConversion(data, 0.2, [2], 1, 0.5, 5, R=True)
    split = int(0.9 * data_corr.shape[0])  # train test split

    # Get all corruption indices of the test set and shift them so that e.g. ind 9050 -> ind 50 on the test set
    corr_ind_test = [i - split for i in corr_ind if i >= split]
    train_corr = data_corr[:split, :]  # Corrupted train-data
    test_corr = data_corr[split:, :]  # Corrupted test-data
    testGT = GT[split:, :]  # Clean test-data

    # Information before applying the models
    print(' Data information:')
    print('  Average JSD:', np.round(Q_JSD(testGT, test_corr), 5))
    print('  Average JSD ocr:', np.round(Q_JSD(testGT[corr_ind_test, :], test_corr[corr_ind_test, :]), 5))
    print('  Average entropy:', Q_H(test_corr))

    # Denoising Autoencoder
    outputs1, _ = run_ae_denoising(testGT, train_corr, test_corr, structure, v=0.30, n=20, e=100, bs=20)
    # BN Model supervised
    path = os.path.join(dir, 'BayesNetwork' + version + '.bif')
    outputs2 = run_bn_sup(path, test_corr, structure)

    results(outputs1, testGT, test_corr, corr_ind_test, 'DAE')
    results(outputs2, testGT, test_corr, corr_ind_test, 'PIBN')
