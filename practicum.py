from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import PandasTools
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np

## read training data

file_in = "data/TR_WS_3158.sdf"
file_out = "cache/TR_WS_3158"+".descr.sdf"

ms = [x for x in Chem.SDMolSupplier(file_in) if x is not None]

ms_wr = Chem.SDWriter(file_out)

nms=[x[0] for x in Descriptors._descList]

calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)

for i in range(len(ms)):
    descrs = calc.CalcDescriptors(ms[i])
    for x in range(len(descrs)):
        ms[i].SetProp(str(nms[x]),str(descrs[x]))
        
    ms_wr.write(ms[i]) 
        
frame = PandasTools.LoadSDF("cache/TR_WS_3158.descr.sdf")       

frame = frame[frame.columns.drop(['ROMol'])]

frame.to_csv('cache/TR_WS_3158.descr.csv')
train = pd.read_csv('cache/TR_WS_3158.descr.csv')
train = train[train.columns.drop(['Unnamed: 0', 'CAS', 'Canonical_QSARr', 'ID', \
                                  'InChI Key_QSARr', 'InChI_Code_QSARr', \
                                  'NAME', 'SMILES', 'Salt_Solvent', 'WS Reference', \
                                  'dsstox_substance_id', 'iupac', 'preferred_name'])]
train = train.dropna()

X_train = train[train.columns.drop(['LogMolar'])]
y_train = train[['LogMolar']]

## read test data

file_in = "data/TST_WS_1066.sdf"
file_out = "cache/TST_WS_1066"+".descr.sdf"

ms = [x for x in Chem.SDMolSupplier(file_in) if x is not None]

ms_wr = Chem.SDWriter(file_out)

nms=[x[0] for x in Descriptors._descList]

calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)

for i in range(len(ms)):
    descrs = calc.CalcDescriptors(ms[i])
    for x in range(len(descrs)):
        ms[i].SetProp(str(nms[x]),str(descrs[x]))
        
    ms_wr.write(ms[i]) 
        
frame = PandasTools.LoadSDF("cache/TST_WS_1066.descr.sdf")       

frame = frame[frame.columns.drop(['ROMol'])]

frame.to_csv('cache/TST_WS_1066.descr.csv')
test = pd.read_csv('cache/TST_WS_1066.descr.csv')
test = test[test.columns.drop(['Unnamed: 0', 'CAS', 'Canonical_QSARr', 'ID', \
                               'InChI Key_QSARr', 'InChI_Code_QSARr', \
                               'NAME', 'SMILES', 'Salt_Solvent', 'WS Reference', \
                               'dsstox_substance_id', 'iupac', 'preferred_name'])]
test = test.dropna()
X_test = test[test.columns.drop(['LogMolar'])]
y_test = test[['LogMolar']]

## identify / remove near-zero variance decriptors

def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

nzv = variance_threshold_selector(X_train, 0.05)

X_train = X_train[nzv.columns]
X_test = X_test[nzv.columns]


## identify / remove highly correlated descriptors

corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
to_drop = [column for column in upper.columns
           if any(upper[column] > 0.85)]

X_train = X_train[X_train.columns.drop(to_drop)]
X_test = X_test[X_test.columns.drop(to_drop)]

## tandardize features by removing the mean and scaling to unit variance

#### StandardScaler: mean=0, variance=1
#### df = preprocessing.StandardScaler().fit_transform(df)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_standard = scaler.transform(X_train)
X_test_standard = scaler.transform(X_train)

## random forest regression

from sklearn.ensemble import RandomForestRegressor
%matplotlib inline
import matplotlib.pyplot as plt

forest = RandomForestRegressor(200)
conv_arr= y_train.values
y_train_array = conv_arr.ravel()
forest.fit(X_train, y_train_array)

y_predict = forest.predict(X_test)

conv_arr= y_test.values
y_test_array = conv_arr.ravel()

fig, ax = plt.subplots()
ax.scatter(y_predict, y_test_array, alpha=0.35)
ax.set_xlim((-12, 2))
ax.set_ylim((-12, 2))
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))
ax.grid(b=True, which='major', linestyle='--')

ax.scatter(x, y, c=colors, alpha=0.5)
ax.set_xlim((0,2))
ax.set_ylim((0,2))
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))
ax.grid(b=True, which='major', color='k', linestyle='--')
fig.savefig('test.png', dpi=600)
plt.close(fig)
