import codecs
import json
import numpy as np
import pandas as pd
import sys
import re
import math

sys.path.append("")


def euclidean_distance(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def gendata1(input_fname):
    '''
    Convert data về phù hợp với bài toán của mình
    '''
    f = open(input_fname, 'r')
    data = f.read().split('\n')
    line = data[0].strip().split(' ')
    # print(line)
    n_vehicles, n_customers, n_depots = int(line[0]), int(line[1]), int(line[2])
    
    res = open(input_fname + '.res', 'r').read().split('\n')
    n_routes = np.zeros(n_depots)
    for i in range(1, len(res)-1):
        # print(res[i])
        # print(int(res[i].split(' ')[0])-1)
        n_routes[int(res[i].split(' ')[0])-1] += 1
    depots = []
    customers = []
    cnt = 1
    for i in range(n_depots):
        capacity = int(data[1+i].strip().split(' ')[1])
        # line = re.sub(' +', ' ', line)
        line = re.sub(' +', ' ', data[1+n_depots+n_customers+i]).strip().split(' ')
        id, x, y = line[0], line[1], line[2]
        depots.append([int(id), '', 'D'+id, 24*3600, float(x), float(y), ' ', 0, capacity])
        for j in range(1, int(n_routes[i])):
            # print(data[-2].strip().split(' ')[0])
            id = int(data[-2].strip().split(' ')[0])
            depots.append([id+cnt, '', 'D' + str(id + cnt), 24*3600, float(x), float(y), ' ', 0, capacity])
            cnt+=1
    
    for i in range(n_customers):
        line = re.sub(' +', ' ', data[1+n_depots+i]).strip().split(' ')
        id, x, y = line[0], line[1], line[2]
        demand = int(line[4])
        customers.append([int(id), '', 'C'+id, 24*3600, float(x), float(y), ' ', 0, demand, 'No'])
    
    
    depot_df = pd.DataFrame(depots, columns=["id","address","code","end_time","latitude","longitude","name","start_time","item_1"])
    depot_df.to_csv(input_fname+'_depot.csv')
    print(f"Depot data in {input_fname + '_depot.csv'}")
    customer_df = pd.DataFrame(customers, columns=["id","address","code","end_time","latitude","longitude","name","start_time","item_1", "seller"])
    customer_df.to_csv(input_fname+'_customer.csv')
    print(f"Customer data in {input_fname + '_customer.csv'}")

    correlation = []
    all_node = depots + customers
    for n1 in all_node:
        for n2 in all_node:
            correlation.append([n1[2], n2[2], euclidean_distance((n1[4], n1[5]), (n2[4], n2[5]))*1000, euclidean_distance((n1[4], n1[5]), (n2[4], n2[5]))])
            correlation.append([n2[2], n1[2], euclidean_distance((n1[4], n1[5]), (n2[4], n2[5]))*1000, euclidean_distance((n1[4], n1[5]), (n2[4], n2[5]))])
    
    pd.DataFrame(correlation, columns=['from_node_code', 'to_node_code', 'distance', 'time']).to_csv(input_fname+'_correlation.csv')
    print(f"Correlation data in {input_fname + '_correlation.csv'}")
    
    print(f"{input_fname}", file=open(r"D:\TaiLieuHocTap\DANO\benchmark_data\p_all.res", 'a'))
    return input_fname + '_depot.csv', input_fname + '_customer.csv', input_fname + '_correlation.csv', n_vehicles, depots[0][-1]
    
def gendata2(cust_fname, depot_fname, corr_fname, n_vehicles, out_dir):
    '''
    Convert từ data của mình sang để đối sánh
    Kịch bản convert: Đổi tọa độ từ lat long về Oxy một cách tương đối: Tìm giá trị nhỏ nhất, lớn nhất của lat và của long
    Sau đó chuẩn hóa về không gian [0, 200] x [0, 200] bằng cách: int(val / min_val * 200)
    Quan trọng nhất là file correlation, tọa độ đã được convert không được sử dụng trực tiếp để tính khoảng cách giữa các node mà chỉ được dùng để truy xuất dữ liệu khoảng cách từ file
    '''
    data = []
    norm_range = 200
    low_val = {'lat': 10000, 'long': 10000}
    high_val = {'lat': 0, 'long': 0}
    
    customers = json.load(codecs.open(cust_fname, 'r', 'utf-8-sig'))
    depots = json.load(codecs.open(depot_fname, 'r', 'utf-8-sig'))
    
    # Tìm các  giá trị low_val và high_val
    for cust in list(customers['market'].values()) + list(depots['depot'].values()):
        if low_val['lat'] > cust['location']['lat']: low_val['lat'] = cust['location']['lat']
        if low_val['long'] > cust['location']['long']: low_val['long'] = cust['location']['long']
        if high_val['lat'] < cust['location']['lat']: high_val['lat'] = cust['location']['lat']
        if high_val['long'] < cust['location']['long']: high_val['long'] = cust['location']['long']
    
    lat_scale = high_val['lat'] - low_val['lat']
    long_scale = high_val['long'] - low_val['long']
    
    id_latlong_map = {} # Lưu lại thông tin ánh xạ id sang latlong
    old_corr = json.load(open(corr_fname, 'r'))
    
    data.append(' '.join([str(n_vehicles), str(len(customers['market'])), str(len(depots['depot']))]))
    for depot in depots['depot'].values():
        data.append(' '.join([str(0), str(depot['demand_list']['1']['demand'])]))
    
    for cust in customers['market'].values():
        lat = int((cust['location']['lat'] - low_val['lat'])/lat_scale * norm_range)
        long = int((cust['location']['long']- low_val['long'])/long_scale * norm_range)
        id_latlong_map[cust['code']] = lat + long/1000
        
        line = [' '.join([str(cust['id']), str(lat), str(long), str(0)]), str(cust['demand_list']['1']['demand'])]
        data.append('  '.join(line))
        
    for depot in depots['depot'].values():
        lat = int((depot['location']['lat'] - low_val['lat'])/lat_scale * norm_range)
        long = int((depot['location']['long'] - low_val['long'])/long_scale * norm_range)
        id_latlong_map[depot['code']] = lat + long/1000
        data.append(' '.join([str(depot['id']), str(lat), str(long)]))
        
    with open(out_dir + '/' + str(len(depots['depot'])) + '_' + str(len(customers['market'])), 'w') as f: 
        f.write('\n'.join(data))
    
    new_corr = {}
    for k1 in id_latlong_map.keys():
        new_corr[id_latlong_map[k1]] = {}
        for k2 in id_latlong_map.keys():
            # print(f"{k1}, {k2}, {id_latlong_map[k1]}, {id_latlong_map[k2]}")
            new_corr[id_latlong_map[k1]][id_latlong_map[k2]] = old_corr[k1][k2]
    
    with open( out_dir + '/new_correlation.json', 'w') as f: 
        json.dump(new_corr, f, indent=4)
def change_config(depot_fname, customer_fname, correlation_fname, n_vehicles, vehicle_capacity, config_fname = 'src/config/config.yaml'):
    '''
    Hàm để thay đổi các thông tin trong file config cho phù hợp với bộ dữ liệu benchmark được convert về kiểu của mình
    '''
    from src.utils.get_data import read_config, dump_config
    config = read_config(config_fname)
    config['gendata']['gen_customer']['fname'] = customer_fname
    config['gendata']['gen_depot']['fname'] = depot_fname
    config['gendata']['gen_correlation']['fname'] = correlation_fname
    config['gendata']['gen_correlation']['flag'] = True
    config['gendata']['gen_vehicle']['n'] = n_vehicles
    config['gendata']['gen_customer']['n'] = -1
    config['gendata']['gen_vehicle']['flag'] = True
    config['gendata']['gen_depot']['n'] = -1
    config['gendata']['gen_depot']['flag'] = True
    config['gendata']['gen_vehicle']['high_thr'] = vehicle_capacity
    config['gendata']['gen_vehicle']['low_thr'] = vehicle_capacity
    config['gendata']['gen_vehicle']['flag'] = True
    config['flag']['depot-customer'] = True
    config['flag']['gendata'] = True
    config['flag']['run_all'] = False
    config['flag']['vendor-depot'] = False
    config['other_config']['distance_type'] = 'euclidean'
    dump_config(config)
    
    
    
if __name__ == '__main__':
    assert len(sys.argv) >= 2, 'specify convert type: 1 (to 2evrp data) or 2 (from 2evrp data)'
    if sys.argv[1] == '1': # Convert data về bài toán 2evrp
        assert len(sys.argv) >= 3, 'specify input data file name'
        depot_fname, customer_fname, correlation_fname, n_vehicles, vehicle_capacity = gendata1(sys.argv[2])
        change_config(depot_fname, customer_fname, correlation_fname, n_vehicles, vehicle_capacity)
        
    if sys.argv[1] == '2': # Convert data từ bài toán 2evrp
        assert len(sys.argv) >= 6, 'Specify some infomation: customer_file(json), depot_file (json), correlation_file (json), n_vehicle, output directory'
        gendata2(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

'''
Scenarios: 
'''
# C:/Users/trung/miniconda3/envs/project2/python.exe "d:/TaiLieuHocTap/Năm 4 - Kỳ 2/ĐATN/source code/port_data/gendata.py" 1 "D:\TaiLieuHocTap\Năm 4 - Kỳ 2\ĐATN\MDVRP_MHA-master\GA\data_official\p03"
