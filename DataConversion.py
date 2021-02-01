import numpy as np
import copy as cp
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


def dataConversion(data, I, attr, Na, noise, alpha, R=True):
    """"
    Method that transforms the data to a one-hot-encoding like structure and adds noise to it so that the data resembles
    probabilistic data.

    :param data: the data that is corrupted, containing for each attribute an integer specifying which category is GT
    :param I: the fraction of records that is corrupted
    :param attr: the specific attributes that are corrupted in case R = False
    :param Na: the number of attributes that are corrupted in case R = False
    :param alpha: the parameter that governs the skewness of the corruption (Dirichlet)
    :param R: whether the indices of the corrupted attributes are randomly selected or not

    :returns GT data (one-hot enc.), corrupted data, structure of data and indices of corrupted records
    """
    n = data.shape[0]
    m = data.shape[1]

    data_dict = {}
    structure = []
    # Transform the data to a dictionary and get the data structure
    for j in range(m):
        temp_data = data[:, j].reshape(-1, 1)
        enc = preprocessing.OneHotEncoder(categories='auto')
        enc.fit(temp_data)
        data_dict[j] = enc.transform(temp_data).toarray()
        structure.append(len(set(data[:, j])))

    x = np.rint(I * n).astype(int)
    # The indices of the records to which noise will be added
    row_inds = np.random.choice(range(n), x, replace=False)
    # For each corrupted observation:
    for i in row_inds:
        if R:
            attr_inds = np.random.choice(m, size=Na, replace=False)
        else:
            # Specify the attributes to which noise will be added
            attr_inds = attr
        # For each specified attribute
        for j in attr_inds:
            # Divide 'noise' over all the columns belonging to that attribute
            noise_individuals = np.random.dirichlet(np.ones(structure[j] - 1) * alpha, size=1) * noise
            t = 0
            for k in range(structure[j]):
                if data_dict[j][i, k] == 0 and t < noise_individuals.shape[1]:
                    data_dict[j][i, k] = noise_individuals[0][t]
                    t += 1
                elif data_dict[j][i, k] == 1:
                    data_dict[j][i, k] -= noise
    new_data = np.concatenate([data_dict[k] for k in data_dict.keys()], axis=1)

    # One-hot encoding GT:
    enc = OneHotEncoder(categories='auto')
    enc.fit(data)
    GT = enc.transform(data).toarray()

    return GT, new_data, structure, row_inds


def noise(data, structure, v):
    """"
    Add noise to data by means of selecting a fraction  v of the attributes and set them to uniform. This is used in a
    Denoising Autoencoder (DAE)
    :param data: the data that is to be corrupted
    :param structure: the structure of the data containing for each attribute the number of categories
    :param v: fraction of attributes that is to be corrupted

    :returns the corrupted data
    """
    # Copy data
    noisy = cp.deepcopy(data)
    # Compute the number of attributes that should be corrupted
    x = int(np.rint(v * len(structure) * data.shape[0]))
    # Pick x random attributes to corrupt
    rand_ind = np.random.choice(len(structure) * data.shape[0], x, replace=False)
    for ind in rand_ind:
        # Determine the exact location of the attribute in the matrix
        i = ind // len(structure)  # record index
        j = ind % len(structure)  # attribute index
        # Set it to a uniform distribution
        left = sum(structure[:j])
        right = left + structure[j]
        noisy[i, left:right] = np.ones(structure[j]) / structure[j]

    return noisy
