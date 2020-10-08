### Instantiate environment
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import PandasTools
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

### Read data
train_df = PandasTools.LoadSDF("data/TR_AOH_516.sdf")
test_df = PandasTools.LoadSDF("data/TST_AOH_176.sdf")

### Concatenate data
AOH = pd.concat([train_df[["Canonical_QSARr", "LogOH"]],
                 test_df[["Canonical_QSARr", "LogOH"]]], ignore_index = True)
AOH['LogOH'] = pd.to_numeric(AOH['LogOH'])

### Calculate features
nms = [x[0] for x in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
for i in range(len(AOH)):
    descrs = calc.CalcDescriptors(Chem.MolFromSmiles(AOH.iloc[i, 0]))
    for x in range(len(descrs)):
        AOH.at[i, str(nms[x])] = descrs[x]
AOH = AOH.dropna()

### Training & Test Datasets
X = AOH.drop(columns=["Canonical_QSARr", "LogOH"])
y = AOH[["LogOH"]]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state = 350,
                                                    test_size = 0.2)

### Identify / remove near-zero variance descriptors
def variance_threshold_selector(data, threshold = 0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices = True)]]

nzv = variance_threshold_selector(X_train, 0.0)

X_train = X_train[nzv.columns]
X_test = X_test[nzv.columns]

### Identify / remove highly correlated descriptors
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                  k = 1).astype(np.bool))
to_drop = [column for column in upper.columns
           if any(upper[column] > 0.85)]

X_train = X_train[X_train.columns.drop(to_drop)]
X_test = X_test[X_test.columns.drop(to_drop)]

### standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaler.fit(X_train)

X_train_standard = scaler.transform(X_train)
X_test_standard = scaler.transform(X_test)

### Random Forest regression


forest = RandomForestRegressor(50)
conv_arr = y_train.values
y_train_array = conv_arr.ravel()
forest.fit(X_train_standard, y_train_array)

y_predict = forest.predict(X_test_standard)

conv_arr= y_test.values
y_test_array = conv_arr.ravel()

fig, ax = plt.subplots()
ax.scatter(y_predict, y_test_array, alpha = 0.35)

y_lower = math.floor(min(y_test_array))
y_upper = math.ceil(max(y_test_array))
x_lower = math.floor(min(y_predict))
x_upper = math.ceil(max(y_predict))
lower = min(x_lower, y_lower)
upper = max(x_upper, y_upper)

x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_xlim((lower, upper))
ax.set_ylim((lower, upper))
ax.set_aspect(abs(x1-x0)/abs(y1-y0))
ax.grid(b=True, which='major', linestyle='--')

init, final = [lower, upper], [lower, upper]

ax.set_title('Random Forest Regression')
ax.set_xlabel('Predicted LogMolar Solubility')
ax.set_ylabel('Observed LogMolar Solubility')

plt.plot(init, final, color = 'red', alpha = 0.5)

#####
##### TPOT
#####

from tpot import TPOTRegressor
tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train_standard, y_train)
print(tpot.score(X_test_standard, y_test))
tpot.export('tpot_AOH_pipeline.py')
