import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append("")
from src.utils.KMeans import * 
from src.utils.get_data import *
from src.utils.SupportClass import *
import matplotlib.pyplot as plt
from src.utils.uitls import *
from src.utils.get_data import *
from src.gendata.gendata import gendata
from src.pipeline.KMeans_phase import KMeans_phase
from src.pipeline.pre_TSP_phase import Pre_TSP_phase
from src.pipeline.TSP_phase import TSP_phase
from src.pipeline.vendor_depot import vendor_tsp_phase
from src.pipeline.survival import sur_main

# Cần fine tune các tham số: penalty_coef, trade_off_coef

def main(dump_flag = True, alpha = 0.9):

    config = read_config('src/config/config.yaml', tpe  = 'yaml')
    n_node_thr, n_customers, n_depots, n_vendor = 0,0,0,0
    n_node_thr = config['other_config']['no of node threshold']
    all_flag = config['flag']
    

    summary = []
    details = []
    route_only = []
    time_calculate = []

    scenarios = ['Scenarios:']
    
    if all_flag['run_all'] or all_flag['gendata']: gendata()

    if all_flag['run_all'] or all_flag['depot-customer']: 
        details.append('1. DEPOT - CUSTOMER PHASE:')
        details.append('1.1. KMeans phase:')
        summary.append('1. DEPOT - CUSTOMER PHASE:')
        summary.append('1.1. KMeans phase:')
        s, d, kmeans_time, n_customers, n_vehicles = KMeans_phase(config['fname']['vehicle'], tpe='depot-customer', alpha=alpha)
        print(f"depot-customer: kmeans done")
        summary.append(s)
        details.append(d)
        scenarios.append('\tNo. of customers: {}'.format(n_customers))
        # time_calculate.append(t)



        details.append('1.2. Prepare for TSP phase:')
        summary.append('1.2. Prepare for TSP phase:')
        s,d, pre_time = Pre_TSP_phase(n_node_thr, config['fname']['vehicle'], tpe='depot-customer', alpha=alpha)
        print(f"depot-customer: pre tsp phase done")
        
        summary.append(s)
        details.append(d)
        # time_calculate.append(t)
        


        details.append('1.3. TSP phase')
        summary.append('1.3. TSP phase')
        s,d, n_depots, tsp_time, r_length, cost = TSP_phase(config['fname']['vehicle'], tpe='depot-customer')
        if s == -1: 
            summary.append('Cannot solving problem, capacity are lower than demand')
            details.append(summary[-1])
            continue_flag = False
        
        else:
            print(f"depot-customer: tsp phase done")
            summary.append(s)
            details.append(d)
            route_only.append('1. DEPOT - CUSTOMER PHASE:')
            route_only.append(d)
            scenarios.append(f"\tNo. of depots: {n_depots}")
            # time_calculate.append(t)
            continue_flag = True




    if (all_flag['run_all'] or all_flag['vendor-depot']) and continue_flag:
        details.append('2. VENDOR - DEPOT PHASE:')
        
        details.append('TSP phase')
        summary.append('TSP phase')
        s, d, n_vendor, t, r_length, cost = vendor_tsp_phase(alpha)
        if s == -1: 
            summary.append('Cannot solving problem, capacity are lower than demand')
            details.append(summary[-1])
        else:    
            print(f"vendor-depot: tsp phase done")
            summary.append(s)
            details.append(d)
            route_only.append('2. VENDOR - DEPOT PHASE:')
            route_only.append(d)
            scenarios.append(f"\tNo. of vendor: {n_vendor}")
            time_calculate.append(t)

    scenarios.append(f"\tNo. of vehicle: {n_vehicles}")
    scenarios.append(f"\tMaximal number of city for TSP: {n_node_thr}")
    summary = scenarios + summary
    details = scenarios + details

    
    # Dump các thông tin ra file:
    if dump_flag:
        scena_fname = f"scenarios/{n_node_thr}node_{n_vehicles}ve_{n_customers}cus_{n_depots}de_{n_vendor}ven"

        if not os.path.exists(scena_fname): os.makedirs(scena_fname)

        
        with open(scena_fname+f"/summary.txt", 'w') as f:
            f.write('\n'.join(summary) + f"\n\nMore details in {scena_fname}/details.txt\n")
            f.write(f'Metric ouput in {scena_fname}/metric_res.json\n')

            f.close()
        with open(scena_fname+f"/details.txt", 'w') as f:
            f.write('\n'.join(details))
            f.close()
        with open(scena_fname+f"/route_only.txt", 'w') as f:
            f.write('\n'.join(route_only))
            f.close()
        
        sur_main(scena_fname+'/metric_res.json')

    # with open('scenarios/parameter_evaluate.csv', 'a') as f: 
    #     f.write(f"{n_node_thr}node_{n_vehicles}ve_{n_customers}cus_{n_depots}de, {r_length}, {cost}, {np.round(kmeans_time + pre_time, 0)}, {tsp_time}\n")
        
    print(f"Folder: {scena_fname}")
    return scena_fname

if __name__ == '__main__': main(alpha=0.9)