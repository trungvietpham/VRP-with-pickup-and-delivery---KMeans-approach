import sys
import os
sys.path.append("")
from src.utils.get_data import *

#csv_to_txt('test_data/test_data_30.csv', 'input/tmp.txt')
def gendata():
    #Đọc các thông tin cấu hình từ file config.txt
    # with open(r'config/config.txt', 'r') as f:
    #     config = f.read()
    # config = config.split('\n')
    # n_types = int(config[0].split('=')[1].strip())

    # Đọc thông tin config:
    config = read_config(r'src/config/config.yaml', tpe='yaml')

    coef_list = config['other_config']['coef list']
    n_types = config['other_config']['no of vehicle types']
    n_items = config['other_config']['no of items']
    n_items_type = config['other_config']['no of items type']
    gen_customer_flag = config['gendata']['gen_customer']['flag']
    gen_vehicle_flag = config['gendata']['gen_vehicle']['flag']
    gen_depot_flag = config['gendata']['gen_depot']['flag']
    gen_vendor_flag = config['gendata']['gen_vendor']['flag']
    n_vehicle = config['gendata']['gen_vehicle']['n']

    if gen_customer_flag: 
        config['fname']['customer'] = config['gendata']['gen_customer']['out_fname']
        csv_to_json_file(config['gendata']['gen_customer']['fname'], data_type='market', dump_file=config['gendata']['gen_customer']['out_fname'], n_node=config['gendata']['gen_customer']['n'], mode='w', low_threshold=config['gendata']['gen_customer']['low_thr'], high_threshold=config['gendata']['gen_customer']['high_thr'], n_items=n_items)
    if gen_vehicle_flag: 
        config['fname']['vehicle'] = 'input/vehicle_{}.json'.format(n_vehicle)
        gen_vehicle(n_vehicle, n_items=n_items_type, dump_file='input/vehicle_{}.json'.format(n_vehicle), low_threshold=config['gendata']['gen_vehicle']['low_thr'], high_threshold=config['gendata']['gen_vehicle']['high_thr'], n_types=n_types, coef_list=coef_list)
    if gen_depot_flag: 
        config['fname']['depot'] = config['gendata']['gen_depot']['out_fname']
        csv_to_json_file(config['gendata']['gen_depot']['fname'], data_type='depot', dump_file=config['gendata']['gen_depot']['out_fname'], mode='w', n_node=config['gendata']['gen_depot']['n'], low_threshold=config['gendata']['gen_depot']['low_thr'], high_threshold=config['gendata']['gen_depot']['high_thr'], n_items=n_items_type)
    if gen_vendor_flag: 
        config['fname']['vendor'] = config['gendata']['gen_vendor']['out_fname']
        csv_to_json_file(config['gendata']['gen_vendor']['fname'], data_type='vendor', dump_file=config['gendata']['gen_vendor']['out_fname'], mode='w', n_node=config['gendata']['gen_vendor']['n'], low_threshold=config['gendata']['gen_vendor']['low_thr'], high_threshold=config['gendata']['gen_vendor']['high_thr'], n_items=n_items)
        gen_vehicle(config['gendata']['gen_vendor']['n'], n_items_type, dump_file='input/vehicle_10.json', low_threshold=config['gendata']['gen_vehicle']['low_thr'], high_threshold=config['gendata']['gen_vehicle']['high_thr'], n_types=n_types, coef_list=coef_list)
        
    get_length_path('data/correlations.csv', 'input/correlation.json')
    get_time_path('data/correlations.csv', 'input/time.json')

    dump_config(config)


if __name__ == '__main__':
    gendata()