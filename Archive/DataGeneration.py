import numpy as np


def generateData2(conditionals, marginals, num_samples):
    """"
    Sample from the joint distribution of (A,B), where A is along the rows, B along the columns
    In the joint matrix, element (i,j) = P(A=a_i,B=b_j)
    :param conditionals: numpy ndarray with P(B|A) conditionals
    :param marginals: numpy array with P(A) marginals
    """
    n = conditionals.shape[0]
    m = conditionals.shape[1]
    joints = np.zeros(conditionals.shape)
    for i in range(n):
        joints[i, :] = conditionals[i, :] * marginals[0, i]
    p = joints.flatten()
    samples_i = np.random.choice(len(p), size=num_samples, p=p)
    samples = [(x // m, x % m) for x in samples_i]
    data = np.zeros((num_samples, 2))
    for i in range(len(samples)):
        data[i, :] = samples[i]
    return data, joints


def generateDataGeneral(factorization, num_samples):
    """"
    Sample from the joint distribution of (X1, X2,..., Xn) via Pomegrenate
    In the joint matrix, element (i,j,...,k) = P(X1=a_i, X2=b_j,...,Xn=c_k)
    :param conditionals: numpy ndarray with
    """
    data = np.zeros((num_samples, len(factorization)))
    i = 0
    for num in range(num_samples):
        ind = []
        for e in factorization:
            if not ind:
                prob = e
                ind.extend(np.random.choice(len(prob), size=1, p=prob))
            elif len(ind) == 1:
                prob = e[ind[0], :]
                ind.extend(np.random.choice(len(prob), size=1, p=prob))
            else:
                prob = e[:, ind[0], ind[1]]
                ind.extend(np.random.choice(len(prob), size=1, p=prob))
        data[i, :] = ind
        i += 1
    return data
