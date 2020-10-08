# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:28:40 2019

@author: US16120
"""

import pandas as pd
import rdkit
from rdkit.Chem import PandasTools
import os
from rdkit import RDConfig

sdfFile = os.path.join('data','TR_AOH_516.sdf')
df = rdkit.Chem.PandasTools.LoadSDF(sdfFile, idName='ID', molColName='ROMol',
                               includeFingerprints=False, isomericSmiles=False,
                               smilesName=None, embedProps=False)
