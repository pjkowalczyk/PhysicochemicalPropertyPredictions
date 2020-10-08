# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 20:23:40 2018

@author: P J Kowalczyk
"""

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
import pandas as pd
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MACCSkeys

sdfile = Chem.SDMolSupplier('data/TR_MP_6486.sdf')

count = 0
for mol in sdfile:
    if mol is None: continue
    count = count + 1
    
print(count)

mol

properties = rdMolDescriptors.Properties()

for name, value in zip(properties.GetPropertyNames(), properties.ComputeProperties(mol)):
  print(name, value)

frame = PandasTools.LoadSDF('data/TR_MP_6486.sdf') 

frame.head()

print(list(frame.head(0)))

# property names in the SDfile
frame.iloc[0,15]

# nms = names of calculated features
nms=[x[0] for x in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)

mol

# calculate features from SMILES string
descriptors = pd.DataFrame(columns = nms)
smiles_strings = frame[frame.columns[1]]
for smiles in smiles_strings:
    descrs = calc.CalcDescriptors(Chem.MolFromSmiles(smiles))
    q = list(descrs)
    qaz = pd.DataFrame(q, nms)
    qaz_transposed = qaz.T
    descriptors = descriptors.append(pd.DataFrame(qaz_transposed))


print(list(descrs))
q = list(descrs)
pd.concat(nms, q)

print(smiles_strings)

descriptors.head

pd.DataFrame([list(descrs)])

descriptors.head()

qaz = pd.DataFrame(q, nms)
qaz
qaz_transposed = qaz.T
qaz_transposed
descriptors = descriptors.append(pd.DataFrame(qaz_transposed))
descriptors

