import math
import numpy as np


def Q_H(data):
    """
    Determines the quality of the data in terms of the relative amount of uncertainty
    Note: the higher the value, the lower the data quality
    :param data: numpy array of shape (observations x variables)
    :return: a measure for the data quality based on information entropy's
    """

    qm = 0
    for e in data:
        qm += -1 * sum([x * math.log2(x) if x else 0 for x in e])
    return qm / data.shape[0]


def Q_KL(data1, data2):
    """"
    Determines the quality improvement of data by means of the Kullback Leibler divergence
    Does so for each observation pair and then takes the average
    :parameter data1: np.ndarray, reference set
    :parameter data2: np.ndarray, predicted set
    """
    eps = 0.01
    total = 0
    n = data1.shape[0]
    m = data1.shape[1]

    for i in range(n):
        total += sum(
            [(data1[i, j] + eps) * math.log((data1[i, j] + eps) / (data2[i, j] + eps)) for j in range(m)])
    return total / n


def Q_JSD(data1, data2):
    """"
    Determines the quality improvement of data by means of the Jensen Shannon divergence
    Does so for each observation pair and then takes the average
    :parameter data1: np.ndarray, reference set
    :parameter data2: np.ndarray, predicted set
    """
    total = 0
    n = data1.shape[0]
    m = data1.shape[1]

    M = 1 / 2 * (data1 + data2)
    for i in range(n):
        total += 1 / 2 * sum(
            [data1[i, j] * math.log(data1[i, j] / M[i, j]) if data1[i, j] != 0 else 0 for j in range(m)]) + 1 / 2 * sum(
            [data2[i, j] * math.log(data2[i, j] / M[i, j]) if data2[i, j] != 0 else 0 for j in range(m)])
    return total / n


# def Q_JSD(data1, data2):
#     data1 = np.clip(data1, 1e-07, 1)
#     data2 = np.clip(data2, 1e-07, 1)
#     total = 0
#
#     n = data1.shape[0]
#     m = data1.shape[1]
#
#     M = 1 / 2 * (data1 + data2)
#     for i in range(n):
#         total += 1 / 2 * sum(
#             [data1[i, j] * math.log(data1[i, j] / M[i, j]) for j in range(m)]) + 1 / 2 * sum(
#             [data2[i, j] * math.log(data2[i, j] / M[i, j]) for j in range(m)])
#
#     return total / n


def Q_faul(GT, corrupted, outputs):
    """"
    Determines for each of the records in the output set whether we have an improvement relative to the corrupted set or not
    """
    faul = 0
    total_sum = 0
    for i in range(GT.shape[0]):
        Q_corr = Q_KL(np.array([GT[i, :]]), np.array([corrupted[i, :]]))
        Q_outp = Q_KL(np.array([GT[i, :]]), np.array([outputs[i, :]]))
        if Q_outp > Q_corr:
            faul += 1
            total_sum += Q_outp - Q_corr
        else:
            continue
    if total_sum == 0:
        return total_sum, 0
    else:
        return faul, total_sum / faul



