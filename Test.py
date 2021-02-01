import os
import pyAgrum as gum
import numpy as np


# def posterior(w, x, structure, alpha, version):
#     """
#     Returns the probability p(x|w)p(w) to compute the best possible correction w
#     :param x: corrupted record
#     :param w: possible correction
#     """
#
#     # Computation is as follows: argmax p(w|x) = argmax p(x|w)p(w)
#     # First determine p(w) via the BN
#     dir = r"C:\Users\Rutger Mauritz\Google Drive\Studie Toegepaste Wiskunde\Module 12\Bachelor Assignment\Datasets" + '\Model' + version
#     bn = gum.loadBN(os.path.join(dir, 'BayesNetwork' + version + '.bif'))
#
#     joint = bn.cpt(list(bn.nodes())[0])
#     for n in list(bn.nodes())[1:]:
#         joint = joint * bn.cpt(n)
#
#     ones = np.array([i for i, x in enumerate(w) if x == 1])
#     ones = ones - np.cumsum(structure[0:3])
#     p_w = joint[ones[0]]
#     for (e, k) in zip(ones[1:], structure[1:]):
#         p_w = p_w[e - k]
#     print('p(w) = ', p_w)
#     p_xw = 0
#     for i in range(2):
#         for j in range(2):
#             for k in range(2):
#                 p_xw += x[i] * x[j + 2] * x[k + 4] * diff(np.array([i, j, k]), ones, alpha)
#                 # print([i, j, k])
#                 # print(x[i] * x[j + 2] * x[k + 4] * diff(np.array([i, j, k]), ones, alpha))
#     print(p_xw)
#     return p_xw * p_w
#
#
# def diff(x, w, alpha):
#     counts = sum(abs(x - w))
#     if counts == 0:
#         return alpha
#     elif counts == 1:
#         return 0.2
#     elif counts == 2:
#         return 0.10
#     elif counts == 3:
#         return 0
#     return (alpha) / (counts + 1)
#
#
# def evidence(x, structure, alpha, version):
#     ev = 0
#     for i in range(2):
#         for j in range(2):
#             for k in range(2):
#                 w = np.zeros(x.shape)
#                 w[i] = 1
#                 w[j + 2] = 1
#                 w[k + 4] = 1
#                 ev += posterior(w, x, structure, alpha, version)
#     return ev
#
#
# def pwx(w, x, structure, alpha, version):
#     return posterior(w, x, structure, alpha, version) / evidence(x, structure, alpha, version)
#
#
# w1 = np.array([1, 0, 1, 0, 0, 1])
# w2 = np.array([1, 0, 1, 0, 1, 0])
# x = np.array([0.9, 0.1, 0.6, 0.4, 0.8, 0.2])
# structure = [0, 2, 2, 2]
#
# print(pwx(w1, x, structure, 1, '7'))
# #print(pwx(w2, x, structure, 1, '7'))
#
# # print(evidence(x, structure, 1, '7'))

from DataConversion import dataConversion
import pandas as pd
version = '2'

dir = r"C:\Users\Rutger Mauritz\Google Drive\Studie Toegepaste Wiskunde\BSc\Module 12\Bachelor Assignment\Datasets" + '\Model' + version
data = pd.read_csv(os.path.join(dir, 'Data' + version + '.csv')).values

GT, data_corr, structure, corr_indices = dataConversion(data, 0.2, [2], 1, 0.5, 5, R=True)
print(corr_indices[:30])
split = int(0.9 * data_corr.shape[0]) # 90% training data, 10% test data
corr_indices = [i - split for i in corr_indices if i >= split]
print("Split = ", split)
print(corr_indices[:10])


train_corr = data_corr[:split, :]  # Corrupted train-data
test_corr = data_corr[split:, :]  # Corrupted test-data
testGT = GT[split:, :]  # Clean test-data


