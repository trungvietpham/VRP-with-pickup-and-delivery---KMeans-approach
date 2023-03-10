
from ast import dump
import codecs
import csv
from dbm import dumb
import random
import numpy as np
import pandas as pd
import json
import sys
import os

import yaml
from yaml.loader import SafeLoader
sys.path.append("")

cnt = 0

def csv_to_json_file(csv_file, n_items = 2, data_type = 'market', dump_file = None, n_node=None, low_threshold = 0, high_threshold = 1000, mode = 'w'):
    '''
    JSON:
    'market': {
        number: {
            'id': 
            'name': 
            'location': {
                'lat': x
                'long': y
            }
            'demand_list':{
                number: {
                    'id': id
                    'demand': demand
                }
            }
        }
    }
    '''
    data_type_list = ['market', 'depot', 'vendor']
    assert data_type in data_type_list, 'data_type must be in {}'.format(data_type_list)
    df = pd.read_csv(csv_file)
    df.reset_index(drop=True, inplace=True)
    # print(df)
    save_data = {}
    # save_data['length'] = len(df)
    markets_dict = {}

    idx = [i for i in range(len(df))]
    k_random = []
    for i in range(n_node):
        r = random.randint(0, len(idx)-1)
        k_random.append(idx[int(r)])
        del idx[r]

    global cnt
    
    for i, line in enumerate(k_random):
        market_dict = {}
        market_dict['id'] = cnt
        market_dict['code'] = df['code'].iloc[line]
        market_dict['name'] = df['name'].iloc[line]
        location_dict = {}
        # if float(df['latitude'][line]) < 18: continue
        location_dict['lat'] = float(df['latitude'].iloc[line])
        location_dict['long'] = float(df['longitude'].iloc[line])
        market_dict['location'] = location_dict
        demands_dict = {}

        for item in range(n_items):
            if 'item_{}'.format(item+1) in df.columns:
                gen_num = df['item_{}'.format(item+1)].iloc[line]
            else: gen_num = random.randint(low_threshold, high_threshold)
            if gen_num!=0:
                demand_dict = {'item_id':item+1, 'demand': gen_num}
                demands_dict[item+1] = demand_dict
        market_dict['demand_list'] = demands_dict

        if 'seller' in df.columns:
            market_dict['seller'] = df['seller'].iloc[line] # Lưu code của người bán (order từ vendor nào)
        
        if 'start_time' in df.columns:
            market_dict['start_time'] = int(df['start_time'][line])
        
        if 'end_time' in df.columns:
            market_dict['end_time'] = int(df['end_time'][line])

        markets_dict[market_dict['id']] = market_dict
        cnt+=1
        
    
    save_data[data_type] = markets_dict
    if not os.path.exists(os.path.dirname(dump_file)): os.makedirs(os.path.dirname(dump_file))
    with open(dump_file, mode, encoding='utf8') as json_file:
        json.dump(save_data, json_file, ensure_ascii=False, indent=4)

def gen_vehicle(n_vehicle, n_items, dump_file, low_threshold = 0, high_threshold = 10000, n_types = 3, coef_list = None):
    save_data = {}
    if coef_list == None:
        coef_list = []
        for _ in range(n_types):
            coef_list.append(round(random.randint(30, 300)/30, 1))
        
    for i in range(n_vehicle):
        vehicle_i = {}
        v_type = random.randint(1, n_types)
        vehicle_i['type'] = v_type
        vehicle_i['coef'] = coef_list[v_type-1]
        for j in range(n_items):
            gen_num = random.randint(low_threshold, high_threshold)
            if gen_num!=0:
                demand_dict = {'item_id':j+1, 'demand': gen_num}
                vehicle_i[j+1] = demand_dict
        save_data[i] = vehicle_i
    
    if not os.path.exists(os.path.dirname(dump_file)): os.makedirs(os.path.dirname(dump_file))
    
    json.dump(save_data, open(dump_file, 'w'), indent=4)

def _mapping_id_code(csv_file):
    df = pd.read_csv(csv_file)
    df.reset_index(drop=True, inplace=True)
    # print(df)
    save_data = {}
    # save_data['length'] = len(df)

    for line in range(len(df)):
        # if float(df['latitude'][line]) < 18: continue
        save_data[df['code'].iloc[line]] = line
    return save_data


def get_time_path(csv_file, dump_file):
    '''
    Lấy ra thông tin về thời gian di chuyển giữa 2 node bất kỳ
    '''
    df = pd.read_csv(csv_file)
    df.reset_index(drop=True, inplace=True)
    customer_map = _mapping_id_code('data/customers.csv')
    depot_map = _mapping_id_code('data/depots.csv')
    node_map = {**customer_map, **depot_map}
    save_data = {}
    for line in range(len(df)):
        from_node_code = str(df['from_node_code'].iloc[line])
        to_node_code = str(df['to_node_code'].iloc[line])
        time = float(df['time'].iloc[line])

        head = from_node_code
        tail = to_node_code
        if head not in save_data: 
            save_data[head] = {}

        save_data[head][tail] = time
    
    json.dump(save_data, open(dump_file, 'w'), indent=4)

def get_length_path(csv_file, dump_file):
    '''
    Lấy ra thông tin về độ dài quãng đường cần di chuyển giữa 2 node bất kỳ
    '''
    
    df = pd.read_csv(csv_file)
    df.reset_index(drop=True, inplace=True)

    customer_map = _mapping_id_code('data/customers.csv')
    depot_map = _mapping_id_code('data/depots.csv')
    node_map = {**customer_map, **depot_map}
    save_data = {}
    for line in range(len(df)):
        from_node_code = str(df['from_node_code'].iloc[line])
        to_node_code = str(df['to_node_code'].iloc[line])
        distance = float(df['distance'].iloc[line])

        head = from_node_code
        tail = to_node_code
        if head not in save_data: 
            save_data[head] = {}

        save_data[head][tail] = distance
    
    json.dump(save_data, open(dump_file, 'w'), indent=4)

def gen_items(n_items, n_type):
    header = ['id', 'name', 'type']
    rows = []
    for i in range(n_items):
        row = [i, 'Hàng {}'.format(i+1), random.randint(0, n_type-1)]
        rows.append(row)
    
    fname = 'data/items.csv'
    with open(fname, 'w', encoding='utf8', newline='') as f:
        write = csv.writer(f)
        write.writerow(header)
        write.writerows(rows)
    
    return fname

def get_items(csv_file):
    '''
    Đọc file csv vào, trả về 1 mảng lưu item type
    '''
    # return pd.read_csv(csv_file)
    data = pd.read_csv(csv_file)
    mapping_type = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        mapping_type[i] = data['type'].iloc[i]
    mapping_type.astype(int)
    return mapping_type

def get_time_load(csv_file):
    '''
    Đọc file csv vào, trả về 1 mảng lưu thời gian load 1 đơn vị hàng hóa
    '''
    # return pd.read_csv(csv_file)
    data = pd.read_csv(csv_file)
    mapping_time = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        mapping_time[i] = data['time'].iloc[i]
    mapping_time.astype(int)
    return mapping_time

def read_config(fname, tpe = 'yaml'):
    '''
    Đọc file config.yaml và trả về 1 dict
    '''

    with codecs.open(fname, 'r', encoding='utf8') as f: 
        if tpe == 'yaml': config = yaml.load(f, Loader=SafeLoader)
        elif tpe == 'json': config = json.load(f)
    return config

def dump_config(config, fname = 'src/config/config.yaml'):
    with codecs.open(fname, 'w', encoding='utf8') as f: 
        if fname[-4:] == 'json': json.dump(config, f, ensure_ascii=False, indent=4)
        elif fname[-4:] == 'yaml': yaml.dump(config, f, indent=4)

def default_config():
    config = {
    'fname': { # luu cac file input cho cac pha
        'correlation': 'input/correlation.json',
        'customer': 'input/market_400.json',
        'depot': 'input/depot_50.json',
        'item_type': 'data/items.csv',
        'vendor': 'input/vendor.json',
        'vehicle': 'input/vehicle_20.json'
    },
    'other_config': {
        'coef list': # coef for corresponse vehicle type
        [7.5,
        2.0,
        5.5,
        4.7,
        9.0],
        'no of items': 5,
        'no of items type': 2,
        'no of vehicle types': 5,
        'no of node threshold': 15 # Number of max city for tsp (depends on your computer, should be in 15-25)
    },
    'output': {
        'depot-customer':{
            'kmeans': 'output/depot-customer/KMeans_phase.json',
            'pre_tsp': 'output/depot-customer/pre_TSP_phase.json',
            'tsp': 'output/depot-customer/TSP_phase_with_Kmeans.json'
        },
        'vendor-depot':{
            'kmeans': 'output/vendor-depot/KMeans_phase.json',
            'pre_tsp': 'output/vendor-depot/pre_TSP_phase.json',
            'tsp': 'output/vendor-depot/TSP_phase_with_Kmeans.json'
        }
    },
    'dump': {
        'customer': 'dump/customer.pkl',
        'depot': 'dump/depot.pkl'
    },
    'flag': { # Flag to specific which file will run
        'run_all': False, # 
        'gendata': False, # Flag to gen data (and then run all)
        'depot-customer': False, # Flag to run all file in phase depot - customer
        'vendor-depot': False # Flag to run all file in phase vendor - depot
    },
    'gendata': {
        'gen_customer': { # Some parameter for gen customer
            'flag': False,
            'fname': 'data/customers.csv',
            'out_fname': None,
            'high_thr': 500,
            'low_thr': 0,
            'n': 400
        },
        'gen_depot': { # Some parameter for gen depot
            'flag': False,
            'fname': 'data/depots.csv',
            'out_fname': None,
            'high_thr': 11000,
            'low_thr': 10000,
            'n': 50
        },
        'gen_vehicle': { # Some parameter for gen vehicle
            'flag': False,
            'fname': None,
            'high_thr': 12000,
            'low_thr': 10000,
            'n': 20
        },
        'gen_vendor': { # Some parameter for gen vendor
            'flag': False,
            'fname': 'data/vendor.csv',
            'high_thr': 20000,
            'low_thr': 15000,
            'n': 10,
            'out_fname': None
        }
    }
    }

    return config
