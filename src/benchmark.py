import matplotlib.pyplot as plt
import numpy as np
# from src.utils.uitls import * 
# from src.utils.KMeans import KMeans
from utils.get_data import * 
import sys
sys.path.append("")
import json
from src.utils.uitls import *
from src.utils.get_data import *
from src.gendata.gendata import gendata
from src.main import main
# Sinh các tập test:

'''
Các trường thông tin cần quan tâm: 
total route length, total cost, total time to calculate kmeans (cluter parent + cluster child), total time TSP
Các trường cần đánh giá:
n_node_thr: 15 -> 20, step 1
n_vehicle: 10 -> 50, step 5
n_customer: 300 -> 425, step 25
n_depot: 10 -> 50, step 10
'''

def gen_data_for_benchmark():
    n_node_list = list(range(15, 25, 1))
    n_vehicle_list = list(range(5, 55, 5))
    n_customer_list = list(range(100, 450, 100))
    n_depot_list = list(range(10, 60, 10))

    default = {
        'n_node_thr': 15,
        'n_vehicle': 20,
        'n_customer': 400,
        'n_depot': 50
    }



    # gen all data for benchmark
    # for n_vehicle in n_vehicle_list:
    #     config = default_config()
    #     config['flag']['gendata'] = True
    #     config['gendata']['gen_vehicle']['flag'] = True
    #     config['gendata']['gen_vehicle']['n'] = n_vehicle
    #     config['gendata']['gen_vehicle']['low_thr'] = int(500*400/n_vehicle)
    #     config['gendata']['gen_vehicle']['high_thr'] = int(config['gendata']['gen_vehicle']['low_thr'] + 2000)
    #     dump_config(config)
    #     gendata()
    # print(f"Gen vehicle done")

    for n_customer in n_customer_list:
        config = default_config()
        config['flag']['gendata'] = True
        config['gendata']['gen_customer']['flag'] = True
        config['gendata']['gen_customer']['n'] = n_customer
        config['gendata']['gen_customer']['out_fname'] = 'input/market_{}.json'.format(n_customer)
        dump_config(config)
        gendata()
    print(f"Gen customer done")

    # for n_depot in n_depot_list:
    #     config = default_config()
    #     config['flag']['gendata'] = True
    #     config['gendata']['gen_depot']['flag'] = True
    #     config['gendata']['gen_depot']['n'] = n_depot
    #     config['gendata']['gen_depot']['out_fname'] = 'input/depot_{}.json'.format(n_depot)
    #     config['gendata']['gen_depot']['low_thr'] = int(500*400/n_depot*2.5)
    #     config['gendata']['gen_depot']['high_thr'] = int(config['gendata']['gen_depot']['low_thr'] + 1000)
    #     dump_config(config)
    #     gendata()
    # print(f"Gen depot done")

def benchmarking():
    '''
    Benchmark thuật toán với các bộ dữ liệu đã được sinh trước, lưu các kết quả vào 1 file 'scenarios/paprameter_evaluate.csv'
    '''
    n_node_list = list(range(10, 20, 1))
    n_vehicle_list = list(range(5, 55, 5))
    n_customer_list = list(range(100, 450, 100))
    # n_depot_list = list(range(10, 60, 10))

    default = {
        'n_node_thr': 15,
        'n_vehicle': 20,
        'n_customer': 400,
        'n_depot': 50
    }

    length = len(n_node_list) + len(n_vehicle_list) + len(n_customer_list)
    cnt = 0

    # for n_thr in n_node_list:
    #     config = default_config()
    #     config['other_config']['no of node threshold'] = n_thr
    #     config['flag']['depot-customer'] = True
    #     config['flag']['vendor-depot'] = True

    #     config['fname']['depot'] = change_fname(config['fname']['depot'], default['n_depot'])
    #     config['fname']['vehicle'] = change_fname(config['fname']['vehicle'], default['n_vehicle'])
    #     config['fname']['customer'] = change_fname(config['fname']['customer'], default['n_customer'])

    #     dump_config(config)
    #     fname, kmean_time, tsp_time = main()
    #     with open('scenarios/parameter_evaluate.csv', 'a') as f: 
    #         f.write(f"thr_{fname}, {kmean_time}, {tsp_time} \n")

    #     cnt+=1 
    #     print(f"Done {cnt}/{length}")
        
    for n_vehicle in n_vehicle_list:
        config = default_config()
        config['flag']['depot-customer'] = True

        config['other_config']['no of node threshold'] = default['n_node_thr']
        config['fname']['depot'] = change_fname(config['fname']['depot'], default['n_depot'])
        config['fname']['vehicle'] = change_fname(config['fname']['vehicle'], n_vehicle)
        config['fname']['customer'] = change_fname(config['fname']['customer'], default['n_customer'])
        dump_config(config)
        fname, kmean_time, tsp_time = main()
        with open('scenarios/parameter_evaluate.csv', 'a') as f: 
            f.write(f"vehicle_{fname}, {kmean_time}, {tsp_time} \n")
        cnt+=1 
        print(f"Done {cnt}/{length}")
    
    for n_customer in n_customer_list:
        config = default_config()
        config['flag']['depot-customer'] = True

        config['other_config']['no of node threshold'] = default['n_node_thr']
        config['fname']['depot'] = change_fname(config['fname']['depot'], default['n_depot'])
        config['fname']['vehicle'] = change_fname(config['fname']['vehicle'], default['n_vehicle'])
        config['fname']['customer'] = change_fname(config['fname']['customer'], n_customer)
        dump_config(config)
        fname, kmean_time, tsp_time = main()
        with open('scenarios/parameter_evaluate.csv', 'a') as f: 
            f.write(f"cust_{fname}, {kmean_time}, {tsp_time} \n")
        cnt+=1 
        print(f"Done {cnt}/{length}")
    
    # for n_depot in n_depot_list:
    #     config = default_config()
    #     config['flag']['depot-customer'] = True

    #     config['other_config']['no of node threshold'] = default['n_node_thr']
    #     config['fname']['depot'] = change_fname(config['fname']['depot'], n_depot)
    #     config['fname']['vehicle'] = change_fname(config['fname']['vehicle'], default['n_vehicle'])
    #     config['fname']['customer'] = change_fname(config['fname']['customer'], default['n_customer'])
    #     dump_config(config)
    #     main(dump_flag=False)
    #     cnt+=1 
    #     print(f"Done {cnt}/{length}")


def alpha_tuning():
    alpha_list = np.arange(0.5, 1.01, 0.05)
    # alpha_list = [1]
    for alpha in alpha_list:
        alpha = round(alpha, 2)
        fname, _, _ = main(alpha=alpha)
        metric = json.load(open(fname+'/metric_res.json', 'r'))
        with open('alpha_eval.csv', 'a') as f:
            f.write(f"{alpha}, \
                    {metric['vendor-depot']['Cost']}, \
                    {metric['vendor-depot']['survivability']}, \
                    {metric['depot-customer']['Cost']}, \
                    {metric['depot-customer']['survivability']}\n")
            f.close()
    
    
if __name__ == '__main__': 
    # gen_data_for_benchmark()
    for i in range(4): 
        benchmarking()
    # for i in range(4): alpha_tuning()