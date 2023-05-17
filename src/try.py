import os
import sys
sys.path.append("")

import pandas as pd
import random
from src.utils.get_data import *
from src.utils.uitls import *

import pandas as pd

data = pd.read_csv(f'data\customers.csv')

data.drop('seller', axis=1, inplace=True)
data['seller'] = ['C5'] * data.shape[0]

data.to_csv(f'data\customers1.csv')