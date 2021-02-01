import pyAgrum as gum
import itertools
import numpy as np


def run_bn_sup(path, test_corr, structure):
    """"
    This method first loads the GT BN, then it propagates evidence from test_corr through it, after
    which a new data set is created based on the new posteriors
    :param path: the path from which the .bif file is loaded
    :param test_corr: test-data that is being updated, in one-hot encoding form
    :param structure: structure of the data (how many categories each attribute has)
    """

    structure_0 = [0] + structure
    # Load the BN
    bn = gum.loadBN(path)
    # Create a placeholder for the net_data
    new_data = np.zeros(test_corr.shape)

    for i in range(test_corr.shape[0]):
        dp = test_corr[i, :]  # fix an observation
        evs = {}
        k = 0
        for n in bn.nodes():  # Convert the evidence to a dictionary structure needed for propagation
            evs[n] = dp[sum(structure_0[:k + 1]):sum(structure_0[:k + 2])]
            k += 1
        ie = gum.LazyPropagation(bn)
        ie.setEvidence(evs)  # set the evidence
        pst = [ie.posterior(n).toarray() for n in bn.nodes()]  # Extract the posteriors and store them in new_data
        new_data[i, :] = list(itertools.chain.from_iterable(pst))
        ie.eraseAllEvidence()
    return new_data
