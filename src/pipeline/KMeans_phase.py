import time
import numpy as np 
import sys
from sklearn import cluster

import os
import sys
# path = os.path.join(os.path.dirname(__file__))
# sys.path.insert(1, path)
sys.path.append("")
from src.utils.KMeans import KMeans
from src.utils.uitls import *
from src.utils.get_data import *

def KMeans_phase(vehicle_fname, tpe = 'depot-customer', alpha = 0.9):
    '''
    Chạy phase phân cụm cho k xe \n
    `tpe`: Loại phân cụm, nhận giá trị là 'depot-customer', 'vendor-depot'\n
    '''
    tpe_list = ['depot-customer', 'vendor-depot']
    assert tpe in tpe_list, 'tpe must be in {}, found {}'.format(tpe_list, tpe)

    # Khởi tạo các thông tin về tên đường dẫn tới các file cần thiết
    config = read_config(r'src/config/config.yaml', tpe='yaml')    
    correlation_fname = config['fname']['correlation']
    item_type_fname = config['fname']['item_type']    
    n_items = config['other_config']['no of items']
    n_items_type = config['other_config']['no of items type']
    (n_vehicles, vehicle_list) = load_vehicle_from_json(vehicle_fname, n_items=n_items)
    correlation = json.load(open(correlation_fname, 'r'))
    mapping_item_type = get_items(item_type_fname)

    if tpe == 'depot-customer':
        
        city_fname = config['fname']['customer']

        # Load các thông tin từ file lên
        (n_cities, city_list) = load_node_from_json(city_fname, format='market', n_items=n_items)

    elif tpe == 'vendor-depot':
        city_fname = config['fname']['depot']

        # Load các thông tin từ file lên
        (n_cities, city_list) = load_data(config['dump']['depot'])
    

    # Check để loại bỏ đi các city có items_array = 0
    i = 0
    while True: 
        if i>= len(city_list): break
        if city_list[i].items_array is None:
            del city_list[i]
        else: i+=1

    n_clusters = n_vehicles
    capa = total_capacity(vehicle_list)  # Đếm ra tổng khối lượng mà tất cả các xe có thể tải được đối với từng loại mặt hàng
    demand = total_demand(city_list, mapping_item_type, n_items_type)
    scaler = np.ceil(np.max(demand/capa)) # Hệ số nhân để tính cluster capacity

    capacity_array = []
    for i in range(n_clusters):
        capacity_array.append(scaler*vehicle_list[i].capacity)
    capacity_array = np.array(capacity_array)

    scale_coef_list = []
    for vehicle in vehicle_list:
        scale_coef_list.append(vehicle.coef)

    # Khởi tạo các cluster:
    cluster_list = []
    for i in range(n_clusters):
        cluster_list.append(Cluster(None, None, capacity=capacity_array[i], scale_coef=scale_coef_list[i]))
        
    # Khởi tạo model
    model = KMeans(n_clusters)

    time1 = time.time()
    if tpe == 'depot-customer': (centers, labels, it, dis, best, i_best, cluster_list, city_list) = model.fit(city_list, cluster_list, correlation, optimizer, mapping_item_type=mapping_item_type, epsilon=5*1e-6, penalty_coef=3, trade_off_coef=alpha, n_times=5)
    if tpe == 'vendor-depot': (centers, labels, it, dis, best, i_best, cluster_list, city_list) = model.fit(city_list, cluster_list, correlation, optimizer, mapping_item_type=mapping_item_type, epsilon=6*1e-6, penalty_coef=3, trade_off_coef=alpha, n_times=5)

    time2 = time.time()

    print('Calculating time: {}'.format(round((time2-time1)*1000, 2)))
    print('Best dis: {}'.format(best))
    print(f"Number of iter: {len(centers)}")

    
    # # plotting data
    # plt.ion()
    # figure, ax = plt.subplots(figsize=(10, 8))

    # figure, ax = draw_animation(figure, ax, centers, model.locations, labels, 0.01, n_vehicles)
    # figure, ax = plot(figure, ax, dis)
    # plt.show()
    # input("Press to continue: ")
    

    # Các biến để in thông tin ra màn hình
    total_mass = np.zeros(n_items_type)
    for cluster in cluster_list:
        total_mass += np.array(cluster.current_mass)

    # In các thông tin ra terminal
    print('\tSummary: ')
    print('\t\tConverged after {} steps'.format(it))
    print('\t\tNo. of clusters = No. of vehicles = {}'.format(len(cluster_list)))
    print('\t\tNo. of customers = {}'.format(len(labels[-1])))
    print('\t\tNo. of good types =  {}'.format(n_items))
    print('\t\tTotal picked capacity = {} kg'.format(total_mass))
    print('\t\tTotal delivered capacity = {} kg'.format(total_mass))
    print('\t\tTotal distance = {} (km)'.format(round(dis[i_best], 3)))
    print('\t\tClustering duration = {} ms'.format(round((time2-time1)*1000.0, 0)))

    # Lưu các thông tin vào file json (để đọc) và 1 file dump (để đưa vào pha khác)

    folder = os.path.dirname(config['output'][tpe]['kmeans'])
    if not os.path.exists(folder): 
            os.makedirs(folder)

    output_to_json_file(cluster_list, city_list, config['output'][tpe]['kmeans'])
    if tpe == 'depot-customer':
        if not os.path.exists(os.path.dirname(config['dump']['customer'])): os.makedirs(os.path.dirname(config['dump']['customer']))
        dump_data(city_list, config['dump']['customer'])
    elif tpe == 'vendor-depot':
        if not os.path.exists(os.path.dirname(config['dump']['depot'])): os.makedirs(os.path.dirname(config['dump']['depot']))
        dump_data(city_list, config['dump']['depot'])
    
    
    summary = []
    details = []
    summary.append('\tSummary: ')
    summary.append('\t\tConverged after {} steps'.format(it))
    summary.append('\t\tNo. of clusters = No. of vehicles = {}'.format(len(cluster_list)))
    summary.append('\t\tNo. of customers = {}'.format(len(labels[-1])))
    summary.append('\t\tNo. of good types =  {}'.format(n_items))
    summary.append('\t\tTotal picked capacity = {} kg'.format(total_mass))
    summary.append('\t\tTotal delivered capacity = {} kg'.format(total_mass))
    summary.append('\t\tTotal distance = {} (km)'.format(round(dis[i_best], 3)))
    summary.append('\t\tClustering duration = {} ms'.format(round((time2-time1)*1000.0, 0)))

    details.append('Description: Clustering {} customers into {} cluster ({} is no. of vehicles) by using KMeans clustering.\n'.format(n_cities, n_clusters, n_vehicles))
    details.append('\n'.join(summary))
    details.append('\tDetails:')

    for i in range(len(cluster_list)):
        details.append('\t\tCluster {}:'.format(i))
        details.append('\t\t\tCurrent mass = {}'.format(cluster_list[i].current_mass))
        details.append('\t\t\tNo. of customers = {}'.format(cluster_list[i].n_cities))
        details.append('\t\t\tCustomers list: {}'.format(cluster_list[i].cities_id))
        details.append('\t\t\tDistance = {} (km)'.format(round(dis[i_best], 3)))
    details.append('\n\n')
    summary.append('\n\n')
    return ('\n'.join(summary), '\n'.join(details), round((time2-time1)*1000.0, 0), n_cities, n_vehicles)
    # # return (summary, details, time in ms)

# if __name__ == '__main__':
#     KMeans_phase('input/vehicle_20.json', tpe='vendor-depot')