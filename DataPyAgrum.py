import pyAgrum as gum
import pyAgrum.lib.bn2graph as viz
import os

# In this file one can create a Bayesian Network via PyAgrum
# The network is then stored

version = '11'
bn = gum.BayesNet('MyNetwork'+version)

# Nodes = Discrete RV's
A = bn.add(gum.LabelizedVariable('A', 'A', 2))
B = bn.add(gum.LabelizedVariable('B', 'B', 2))
C = bn.add(gum.LabelizedVariable('C', 'C', 2))

# Edges = Conditional relationships
for link in [(A, B), (A, C)]:
    bn.addArc(*link)

# Conditional Probability Tables
bn.cpt(A)[:] = [1 / 2, 1 / 2]
bn.cpt(B)[:] = [[1, 0], [0, 1]]
bn.cpt(C)[:] = [[1, 0], [0, 1]]

# Save data and model architecture to separate folder
dbg = gum.BNDatabaseGenerator(bn)
dbg.drawSamples(10000)

dir = 'Datasets'
os.chdir(dir)
dir += '\Model' + version
os.mkdir('Model' + version)

print(os.path.join(dir, 'Data' + version + '.csv'))

#dbg.toCSV(os.path.join(dir, 'Data' + version + '.csv'))
viz.pngize(bn, os.path.join(dir, 'BayesNetwork' + version))
#gum.saveBN(bn, os.path.join(dir, 'BayesNetwork' + version + '.bif'))
