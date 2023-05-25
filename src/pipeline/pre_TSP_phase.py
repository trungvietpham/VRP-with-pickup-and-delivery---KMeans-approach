'''
Cắt cụm to ra thành các cụm con để sử dụng thuật toán TSP
1. Đọc các dữ liệu từ output phase 1 vào và các dữ liệu về xe, items, ...
2. KMeans để chia ra 1 cụm thành các cụm con có thể di chuyển trong 1 route
3. Dump lại các cụm con ra 1 file json
'''

import time
import numpy as np
import json
import sys
import os
sys.path.append("")
from src.utils.SupportClass import Cluster
from src.utils.KMeans import KMeans
from src.utils.uitls import *
from src.utils.get_data import *

def Pre_TSP_phase(n_node_threshold, vehicle_fname, tpe='depot-customer', alpha = 0.9):
    '''
    `n_node_threshold`: Ngưỡng chặn trên cho số node trong 1 cụm con\n
    `vehicle_fname`: đường dẫn tới file lưu thông tin vehicles\n
    `tpe`: Loại phân cụm, nhận giá trị là 'depot-customer', 'vendor-depot'\n
    '''
    tpe_list = ['depot-customer', 'vendor-depot']
    assert tpe in tpe_list, 'tpe must be in {}, found {}'.format(tpe_list, tpe)

    # Khởi tạo các thông tin về tên đường dẫn tới các file cần thiết
    config = read_config(r'src/config/config.yaml', tpe='yaml')
    if tpe == 'depot-customer': city_fname = config['fname']['customer']
    elif tpe == 'vendor-depot': city_fname = config['fname']['depot']
    correlation_fname = config['fname']['correlation']
    item_type_fname = config['fname']['item_type']

    
    # print(f"Double check: Vehicle fname: {vehicle_fname} vs {config['fname']['vehicle']}")
    # input('Continue ...')
    # Load các thông tin từ file lên
    n_items = config['other_config']['no of items']
    n_items_type = config['other_config']['no of items type']
    if tpe == 'depot-customer':
        (n_cities, city_list) = load_data(config['dump']['customer'])
    elif tpe == 'vendor-depot':
        (n_cities, city_list) = load_data(config['dump']['depot'])
    (n_vehicles, vehicle_list) = load_vehicle_from_json(vehicle_fname, n_items=n_items)
    correlation = json.load(open(correlation_fname, 'r'))
    mapping_item_type = get_items(item_type_fname)

    output_phase1_fname = config['output'][tpe]['kmeans']
    dump_file = config['output'][tpe]['pre_tsp']

    # convert_coef = get_convert_coef_from_file(convert_coef_fname)
    cluster_data = json.load(open(output_phase1_fname, 'r'))
    n_clusters = len(cluster_data)
    n_items = vehicle_list[0].capacity.shape[0]

    summary = []
    details = []

    details.append('Description: Clustering {} cluster obtain from previous step into smaller cluster (sub-cluster) for TSP in next step\n'.format(n_vehicles))
    
    '''
    1. Khôi phục lại các cụm từ output_phase1_file
    '''
    print('Recover data')
    # Mapping customer code với id
    mapping_code_id = {}
    mapping_id_idx = {}
    for i, city in enumerate(city_list):
        mapping_code_id[city.id] = city.id
        mapping_id_idx[city.id] = i
    


    cluster_list = []

    scale_coef_list = []
    for i in range(n_vehicles):
        scale_coef_list.append(vehicle_list[i].coef)
    cnt = 0
    for cluster_id in cluster_data:
        x, y = cluster_data[cluster_id]['center']['lat'], cluster_data[cluster_id]['center']['long'] # Tọa độ của tâm cụm cha
        n_cities_i = len(cluster_data[cluster_id]['node_list'])
        child_list = []
        mass = np.zeros(n_items)

        cnt = 0
        for id in cluster_data[cluster_id]['node_list']:
            child_list.append(int(mapping_code_id[int(id)]))
            for i in range(n_items):
                mass[i]+=cluster_data[cluster_id]['node_list'][id]["demand"]['Item '+str(i)]
        
        cluster_list.append(Cluster(x,y,None, n_cities=n_cities_i, cities_id=child_list, current_mass=mass, scale_coef=scale_coef_list[cnt]))
        cnt+=1

    '''
    2. Với mỗi cụm ta chia thành các cụm nhỏ, số lượng cụm nằm trong khoảng [max(ceil(tổng/capa)), n_node/(floor(min(capa/mean)))]
    Số node trong cụm được giới hạn bởi 1 biến, đặt = 15
    '''
    print('Split cluster')
    # Một số biến lưu trữ các thông tin về số lần thử kmeans, số cụm cha, số cụm con
    try_kmeans_counter_list = []
    n_cluster_parent = 0
    n_cluster_child = []
    total_distance = []
    time_computing = []
        
    #Với cụm thứ i, low_n_cluster[i] là chặn dưới số cụm con được phân ra từ cụm cha đó
    #Tiến hành chia cụm và sau đó lưu vào 1 dict để dump vào file
    save_data = {}
    for cluster_id in range(n_clusters):
        n_cluster_parent +=1
        # n_child = 1

        #Lấy ra các node nằm trong cluster cha này
        child_city_list = []
        child_id = []
        child_id = sorted(cluster_list[cluster_id].cities_id)

        for city in child_id:
            child_city_list.append(city_list[mapping_id_idx[city]])

        print(f"Total demand: {total_demand(child_city_list, mapping_item_type, n_items)}, vehicle capacity: {vehicle_list[cluster_id].capacity}")
        n_child = max(int(np.max(np.floor(total_demand(child_city_list, mapping_item_type, n_items)/vehicle_list[cluster_id].capacity))), 1)
        
        # Bắt đầu lặp từ giá trị n_child, mỗi khi phân cụm xong, ta kiểm tra xem 
        # các current_mass có đều nhỏ hơn capacity của xe hay không, nếu không thì ta tăng giá trị n_child và lặp lại
        continue_flag = True
        time1 = time.time()
        try_kmeans_counter = 1
        while continue_flag:
            
            capacity_array = np.array([list(vehicle_list[cluster_id].capacity) for i in range(n_child)]).reshape((n_child, n_items))
            child_scale_coef = [scale_coef_list[cluster_id] for _ in range(n_child)]

            # Khởi tạo các cụm con
            child_cluster_list = []
            for i in range(n_child):
                child_cluster_list.append(Cluster(None, None, capacity_array[i], scale_coef=child_scale_coef[i]))
            
            # print(capacity_array)
            # input("..")
            
            # Khởi tạo model
            model = KMeans(n_child, distance_type=config['other_config']['distance_type'])
            
            # Kiểm tra mỗi node xem demand có lớn hơn capacity của xe hay không: 
            for i, node in enumerate(child_city_list):
                # print(f"Node {i}: demand: {convert_demand(node.items_array, mapping_item_type)}\nVehicle: {cluster_id}, capacity: {vehicle_list[cluster_id].capacity}")
                if not is_bigger_than(vehicle_list[cluster_id].capacity, convert_demand(node.items_array, mapping_item_type)):
                    print(f"Node {i}: demand: {convert_demand(node.items_array, mapping_item_type)}\nVehicle: {cluster_id}, capacity: {vehicle_list[cluster_id].capacity}")
                    print(f"Node demand exceed vehicle capacity, can't clustering")
                    return -1, -1, -1

            (centers, labels, it, dis, best, i_best, child_cluster_list, child_city_list) = model.fit(child_city_list, child_cluster_list, correlation, optimizer, mapping_item_type, epsilon=5*1e-6, penalty_coef=3, trade_off_coef=alpha, n_times=3)

            continue_flag = False
            for child in child_cluster_list:

                #Kiểm tra nếu current_mass của cụm lớn hơn capacity của xe thì set flag thành true
                if not is_bigger_than(child.capacity, child.current_mass): 
                    continue_flag = True
                    # print(f"Not bigger")
                    
                    break

                #Kiểm tra nếu số node lớn hơn n_node_threshold thì set flag thành true
                # print(f"child: {child.print(city_id_list_flag=True)}, n child: {child.n_cities}, thr: {n_node_threshold}")
                if child.n_cities > n_node_threshold:
                    continue_flag = True

                    break
            
            if continue_flag == True: 
                n_child+=1
                try_kmeans_counter+=1
                del child_cluster_list
                del model
                del capacity_array
            else: 
                time2 = time.time()

                cluster_parent_info = {}
                cluster_parent_info['cluster_id'] = cluster_id
                cluster_parent_info['center'] = cluster_data[str(cluster_id)]['center']

                cluster_children_info = output_to_json_file(child_cluster_list, child_city_list, output_flag=False)

                cluster_parent_info['child_cluster_list'] = cluster_children_info
                save_data[cluster_id] = cluster_parent_info

                # Update các thông tin để in ra màn hình
                n_cluster_child.append(n_child)
                total_distance.append(dis[i_best])
                time_computing.append(time2-time1)
                try_kmeans_counter_list.append(try_kmeans_counter)

    print('Cluster into child done')
    '''
    3. Dump ra file json, lấy tên là pre_TSP_phase.json
    '''
    folder = os.path.dirname(dump_file)
    if not os.path.exists(folder): 
            os.makedirs(folder)

    with open(dump_file, 'w', encoding='utf-8') as json_file:
        json.dump(save_data, json_file, ensure_ascii=False, indent=4)

        #TODO: lấy ra các thành phố thuộc vào cụm cha này, sau đó fit vào model, lưu lại các thông tin về nhãn của từng cụm con để biểu diễn dữ liệu

    print('\tSummary: ')
    print('\t\tTotal try KMeans times = {}'.format(np.sum(try_kmeans_counter_list)))
    print('\t\tNo. cluster parent = {}'.format(n_cluster_parent))
    print('\t\tTotal no. cluster child = {}'.format(np.sum(np.array(n_cluster_child))))
    print('\t\tTotal distance = {} (km)'.format(round(np.sum(np.sum(total_distance)), 0)))
    print('\t\tTotal time for clustering = {} ms'.format(round(np.sum(np.array(time_computing))*1000.0, 0)))

    summary.append('\tSummary: ')
    summary.append('\t\tTotal try KMeans times = {}'.format(np.sum(try_kmeans_counter_list)))
    summary.append('\t\tNo. cluster parent = {}'.format(n_cluster_parent))
    summary.append('\t\tTotal no. cluster child = {}'.format(np.sum(np.array(n_cluster_child))))
    summary.append('\t\tTotal distance = {} (km)'.format(round(np.sum(np.sum(total_distance)), 0)))
    summary.append('\t\tTotal time for clustering = {} ms'.format(round(np.sum(np.array(time_computing))*1000.0, 0)))

    details.append('\n'.join(summary))
    details.append('\tDetails: ')

    for i in range(n_clusters):
        details.append('\t\tCluster parent {}'.format(i))
        details.append('\t\t\tTry KMeans {} times'.format(try_kmeans_counter_list[i]))
        details.append('\t\t\tNo. cluster child: {}'.format(n_cluster_child[i]))
        details.append('\t\t\tTotal distance in all cluster child: {} (km)'.format(round(total_distance[i], 0)))
        details.append('\t\t\tTime for clustering: {} ms'.format(round(time_computing[i]*1000.0, 0)))
    details.append('\n\n')
    summary.append('\n\n')
    return ('\n'.join(summary), '\n'.join(details), round(np.sum(np.array(time_computing))*1000.0, 0))

if __name__ == '__main__':
    Pre_TSP_phase(15, 'input/vehicle_20.json', tpe='depot-customer')