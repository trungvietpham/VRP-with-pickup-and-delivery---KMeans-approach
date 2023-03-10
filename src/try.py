import os
import sys
sys.path.append("")

import pandas as pd
import random
from src.utils.get_data import *
from src.utils.uitls import *

config = read_config('src/config/config.yaml')
# n_v, vendor_list = load_data(config['dump']['depot'])
# for v in vendor_list:
#     # print(f"{v.order_dict}")
#     print(v.order_dict)
#     input('P')

