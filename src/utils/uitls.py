import codecs
import csv
import json
import pickle
import random
import time
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import sys

import yaml
sys.path.append("")
from src.utils.SupportClass import *
def distance(point1, point2) -> float:
    '''
    Tính khoảng cách l2 giữa 2 điểm tọa độ lat long, trả về khoảng cách km
    '''
    R = 6378.137; # Radius of earth in KM
    dLat = (point2[0] - point1[0])*np.pi/180
    dLon = (point2[1]-point1[1])*np.pi/180
    a = np.sin(dLat/2) ** 2 + np.cos(point1[0]*np.pi/180) * np.cos(point2[0]* np.pi / 180) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R*c
    return d

def optimizer(clusters_capacity, nodes_demand, locations, centroid, penalty_coef: int, scale_coef, trade_off_coef, item_type_mapping):
    '''
    Return array [n_nodes, n_clusters]\n
    Hàm tối ưu, với capacity và demand đã được chuẩn hóa về (0,1)\n
    Danh sách các tham số:\n
    `clusters_capacity`:mảng lưu sức chứa của từng cụm\n
    `nodes_demand`: Mảng lưu demand của các node\n
    `locations`: tọa độ của các nodes\n
    `centroid`: tọa độ hiện tại của các tâm cụm\n
    `penalty_coef`: Hệ số phạt trong trường hợp vượt quá sức chứa của cụm\n
    `scale_coef`: Mảng lưu hệ số (chi phí) của xe ứng với các cụm\n
    `trade_off_coef`: Hệ số alpha giữa khoảng cách và demand, thỏa mãn alpha trong khoảng (0,1)\n
    `item_type_mapping`: mảng mapping item và item type\n
    Công thức hàm: alpha * L2(city_i, center_j) - (1-alpha)*(chuyên chở - trọng số)\n
    '''
    # Cast data sang array
    clusters_capacity = np.array(clusters_capacity)
    nodes_demand = np.array(nodes_demand)

    # Kiểm tra dữ liệu đầu vào
    assert 0<trade_off_coef<1, 'trade_off_coef must be an float in range (0,1)'

    # Khởi tạo 1 vài tham số
    zeros_penalty = 1000
    n_clusters, n_items_type = clusters_capacity.shape
    n_cities, n_items = nodes_demand.shape
    current_mass = np.array([[0.0]*n_items_type] * n_clusters)
    res = [] # Lưu lại kết quả hàm tối ưu của mỗi node đối với mỗi tâm cụm
    check_zeros_value = (clusters_capacity == 0) # Kiểm tra xe tại ô nào có giá trị = 0 thì ta hiểu đó là item mà xe được gắn cho cluster đó không chở được, vì thế sẽ đặt zero_penalty vào
    epsilon = 1e-6

    # Lặp qua tất cả dữ liệu nodes demand:
    for i, demand in enumerate(nodes_demand):
        # Tính giá trị về khoảng cách:
        dis = []
        for j, center in enumerate(centroid):
            adis = distance(locations[i], centroid[j])
            dis.append(adis)
        dis = np.array(dis)
        # Tính giá trị về khối lượng:
        adding_mass = convert_demand(nodes_demand[i], item_type_mapping)

        current_mass += adding_mass
        remain_mass = clusters_capacity - current_mass
        check_penalty = (remain_mass<=0)
        mass = scale_coef * np.sum((penalty_coef * check_penalty + (1-check_penalty)) * remain_mass / n_items_type * (zeros_penalty * check_zeros_value + (1-check_zeros_value)), axis=1)
        # Lưu lại kết quả để return 

        max_dis = np.max(dis)
        if max_dis == 0: max_dis = epsilon
        max_mass = np.max(mass)
        if max_mass == 0: max_mass = epsilon
        
        res.append(trade_off_coef * dis / max_dis - (1 - trade_off_coef) * mass / max_mass)

        # Cập nhật lại current_mass: 
        current_mass -= adding_mass
        current_mass[int(np.argmin(res[-1]))] += adding_mass
    
    return np.array(res)

def load_node_from_json(file_name, format, n_items=0) -> tuple[int, list[Node]]:
    '''
ss
    Params: 

    file_name: đường dẫn tới file json

    format: phần nội dung sẽ đọc vào, nhận giá trị là 'market', 'vendor', 'depot'

    định dạng file text: 

    line 1: số lượng thành phố n

    2n line tiếp theo: dòng đầu là tọa độ, dòng sau là demand đối với mỗi loại mặt hàng

    Return:

    n_city: số lượng thành phố

    city_list: list class City
    '''
    format_list = ['market', 'vendor', 'depot']
    if format not in format_list: 
        raise Exception("format must be in list {}. Found {}".format(format_list, format))
    
    f = codecs.open(file_name, 'r', 'utf-8-sig')
    data = json.load(f)
    for key in data[format]: 
        n_items = len(data[format][key]['demand_list'])
        break
    n_cities = len(data[format])
    city_list = []
    for node in data[format]:

        id_i = data[format][node]['id']
        code_i = data[format][node]['code']
        name_i = data[format][node]['name']
        location_i = data[format][node]['location']
        demand_i = data[format][node]['demand_list']
        if 'seller' in data[format][node]:
            seller_i = data[format][node]['seller']
        else: seller_i = None

        if 'start_time' in data[format][node]:
            start_time = data[format][node]['start_time']
        else: start_time = 0

        if 'end_time' in data[format][node]:
            end_time = data[format][node]['end_time']
        else: end_time = 3600 * 24

        # index_i = np.array(re.split(re.compile(' +'), data[3*i+2 + offset]))
        # demand_i = np.array(re.split(re.compile(' +'), data[3*i+3 + offset]))

        demand_list_i = np.zeros(n_items)
        for j in demand_i:
            demand_list_i[demand_i[j]['item_id'] - 1] = float(demand_i[j]['demand'])
        if format == 'market': 
            tpe = 'CUSTOMER'
            city_list.append(Node(location_i['lat'], location_i['long'], id = id_i, code=code_i, name=name_i, tpe=tpe, items_array=demand_list_i, cluster_id=None, seller=seller_i, start_time=start_time, end_time=end_time))
        elif format == 'depot': 
            tpe = 'DEPOT'
            city_list.append(Node(location_i['lat'], location_i['long'], id = id_i, code=code_i, name=name_i, tpe=tpe, items_type_array=demand_list_i, cluster_id=None, remain_capacity=demand_list_i, start_time=start_time, end_time=end_time))
        elif format == 'vendor': 
            tpe = 'VENDOR'
            city_list.append(Node(location_i['lat'], location_i['long'], id = id_i, code=code_i, name=name_i, tpe=tpe, items_array=demand_list_i, cluster_id=None, start_time=start_time, end_time=end_time))
    
    return (n_cities, city_list)

def load_vehicle_from_json(file_name, n_items) -> tuple[int, list[Vehicle]]:
    data = json.load(open(file_name, 'r'))
    n_vehicles = 0
    for key in data:
        n_items = len(data[key]) - 2
        break
    vehicle_list = []
    id = 0
    for key in data:
        n_vehicles+=1
        data_i = np.zeros(n_items)
        v_type = ''
        v_coef = 0.0
        for j in data[key]:
            if j == 'type':
                v_type = data[key][j]
            elif j == 'coef': v_coef = round(float(data[key][j]),1)
            else:
                data_i[int(j)-1] = data[key][j]['demand']
        
        vehicle_list.append(Vehicle(id = id, capacity= data_i, v_type=v_type, coef=v_coef))
        id+=1
    
    return (n_vehicles, vehicle_list)

def total_capacity(vehicle_list: list[Vehicle]):
    '''
    Hàm tính tổng chuyên chở của các xe mỗi loại
    Input: list class Vehicle
    Output: mảng dài n_items
    '''
    n_vehicles = len(vehicle_list)
    total = np.zeros(len(vehicle_list[0].capacity))
    for vehicle in vehicle_list:
        total+=vehicle.capacity
    
    return total

def total_demand(city_list: list[Node], mapping_item_type = None, n_item_type = 2):
    '''
    Hàm tính tổng cần giao của các điểm mỗi loại\n
    Input: \n
    list class City\n
    `mapping_item_type`: mảng để convert item thành item_type, `None` nếu không cần convert\n
    `n_item_type`: số lượng item_type\n
    Output: mảng dài item_type\n
    '''
    total = np.zeros(n_item_type)
    n_item = len(city_list[0].items_array)
    for city in city_list:
        if mapping_item_type is not None:
            # print(city.items_array)
            # input('ppp')
            total += convert_demand(city.items_array, mapping_item_type)
        else: total += city.items_array
    
    return total

def output_to_json_file(cluster_list:list[Cluster], city_list:list[Node], dump_file = 'output/phase2.json', output_flag = True):
    '''
    dạng json: 
    "": {
        id
        center: {
            'lat':x
            'long':y
        }
        node_list:{
            "":{
                id:
                demand:{
                    item1:{
                        name:
                        quantity:
                        unit:
                    }
                }
            }
        }
    }
    '''
    save_data = {}
    n_cluster = len(cluster_list)
    n_city = len(city_list)
    # Mapping city_id ra chỉ số của mảng city_list:
    mapping = {}
    for j in range(n_city):
        mapping[int(city_list[j].id)] = j

    mapping_id_code = {}
    for i in range(len(city_list)):
        mapping_id_code[city_list[i].id] = city_list[i].code
    
    # print('Mapping: ')
    # print(mapping)
    for i in range(n_cluster):
        tmp = {}
        tmp['cluster_id'] = i
        center_tmp = {}

        center_tmp['lat'] = cluster_list[i].lat
        center_tmp['long'] = cluster_list[i].lng
        tmp['center'] = center_tmp

        cities_tmp = {}
        for city_id in cluster_list[i].cities_id:
            city_tmp = {}
            city_tmp['node_id'] = int(city_id)
            # print('City id = {}'.format(city_id))
            # print('Mapping = {}'.format(mapping[city_id]))
            city_tmp['node_location'] = {'lat': city_list[mapping[int(city_id)]].lat, 'long':city_list[mapping[int(city_id)]].lng}

            demand = city_list[mapping[int(city_id)]].items_array
            demand_tmp = {}
            for j in range(len(demand)):
                demand_tmp['Item ' + str(j)] = demand[j]
            
            city_tmp['demand'] = demand_tmp
            # cities_tmp[mapping_id_code[int(city_id)]] = city_tmp
            cities_tmp[city_id] = city_tmp
        tmp['node_list'] = cities_tmp
        save_data[str(i)] = tmp
    
    if output_flag:
        with open(dump_file, 'w', encoding='utf-8') as json_file:
            json.dump(save_data, json_file, ensure_ascii=False, indent=4)
    
    return save_data

def visualize(): pass

def kmeans_display(X, center, label, K, no_color_flag = False):
    if not no_color_flag:
        color_list = ['g', 'b','c', 'm', 'y', 'orange']
        marker_list = ['.', 'v', 'o', 's','p', 'P', '*','+','x']
    else: 
        color_list = ['w'] * 6
        marker_list = ['.']*9
    c_color = []
    c_marker = []
    for i in range(K):
        color = color_list[random.randint(0, len(color_list)-1)]
        marker = marker_list[random.randint(0, len(marker_list)-1)]
        c_color.append(color)
        c_marker.append(marker)
        Xi = X[label==i, :]
        plt.scatter(Xi[:, 0], Xi[:, 1], color = color, marker = marker, s=10, alpha = .8)
        
    
    X_gray = X[label==-1, :]
    plt.scatter(X_gray[:, 0], X_gray[:, 1], color = 'gray', marker = 'o', s = 10, alpha = .8)

    for i in range(K):
        plt.scatter(center[i, 0], center[i, 1], color = 'red', marker = c_marker[i], s = 50, alpha = .8)
    # plt.axis('equal')
    # plt.plot()

def total_distance(centroid, locations, labels):
    '''
    Tính khoảng cách từ các tâm cụm tới các node trong từng cụm.

    Tham số:\n
    `centroid`: tọa độ các tâm\n
    `locations`: tọa độ các node\n
    `labels`: nhãn các node\n

    '''
    centroid = np.array(centroid)
    locations = np.array(locations)
    res = np.zeros(centroid.shape[0])
    for i in range(locations.shape[0]):
        res[labels[i]] += distance(centroid[labels[i]], locations[i])
    
    return np.sum(res)

def draw_animation(figure, ax, all_centroid, locations, all_labels, time_delta, n_clusters, next_by_key = False):
    it = len(all_centroid)
    # plotting data
    plt.ion()
    # figure, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(it):
        kmeans_display(locations, all_centroid[i], all_labels[i], n_clusters)
        
        # plt.show()
        figure.canvas.draw()
        figure.canvas.flush_events()
        # input('A key')
        # plt.waitforbuttonpress(10)
        if next_by_key: input('Press to continue...')
        else:
            time.sleep(time_delta) 
        plt.clf()
    
    return figure, ax

def plot(figure, ax, dis):

    plt.plot(dis, 'blue')
    plt.plot(dis, 'ro')
    n_point = len(dis)
    diff = n_point / 10
    for i in range(10):
        x = int(np.round(i*diff))
        if x>= len(dis): x = x-1
        plt.text(x, dis[x]+10, str(np.round(dis[x])), ha='center')
    plt.text(n_point - 1, dis[-1]+10, str(np.round(dis[x])), ha='center')
    return figure, ax

def convert_demand(demand, mapping_item_type):
    '''
    Convert demand từ 1 list item thành item_type\n

    '''

    n_item_type = int(np.max(mapping_item_type)) + 1
    res = np.array([0.0]*n_item_type)
    for j in range(n_item_type):
        # print(nodes_demand[i][item_type_mapping == j])
        # print(demand[mapping_item_type == j])
        # input('pp')
        res[j] += np.sum(demand[mapping_item_type == j])
    
    return res


def save_data_to_cluster_customer(cluster_list: list[Cluster], center_list, labels_list, customer_list: list[Node], mapping_item_type):
    '''
    Lưu các thông tin sau khi chạy xong kmeans vào các thuộc tính của cluster, customer\n
    Kết quả trả về: cluster_list, customer_list\n
    Input: \n
    `cluster_list`: Danh sách các cluster\n
    `center_list`: Thông tin về tọa độ các clusters\n
    `labels_list`: Thông tin về nhãn của các customers\n
    `customer_list`: Danh sách các customer\n
    '''

    # Cast các thuộc tính thành array
    center_list = np.array(center_list)
    labels_list = np.array(labels_list)

    # print('Labels list: {}'.format(labels_list))

    # Lưu các thuộc tính cho các clusters
    for i, cluster in enumerate(cluster_list):
        city_list = np.where(labels_list == i)[0]
        cluster.set_center(center_list[i])
        # print('Lưu thuộc tính cho cluster và customer: ')
        # print(city_list)
        cluster.clear_mass()
        for j in city_list:
            # print('Customer demand: {}'.format(customer_list[j].items_array))
            # print('Update mass: {}'.format(convert_demand(customer_list[j].items_array, mapping_item_type)))
            # input('PpP')
            cluster.update_mass(convert_demand(customer_list[j].items_array, mapping_item_type), customer_list[j].id)
            # print('Cluster mass after update: {}'.format(cluster.current_mass))
            # input('Press to continue...')
    
    for i, customer in enumerate(customer_list):
        customer.cluster_id = labels_list[i]
    
    return (cluster_list, customer_list)

def is_bigger_than(array1, array2):
    '''
    Kiểm tra xem array1 có lớn hơn array2 hay không, định nghĩa a1>a2 khi và chỉ khi a1[i]>=a2[i] với mọi i
    '''
    array1 = np.array(array1)
    array2 = np.array(array2)

    #Kiểm tra có cùng shape hay không, nếu không thì raise lỗi 
    if array1.shape != array2.shape: 
        raise Exception('2 array do not have the same shape')
    # print('Array 1: {}'.format(array1))
    # print('Array 2: {}'.format(array2))
    # print('bigger:  {}'.format(array1>=array2))
    # print('Return: {}'.format(np.sum(array1>=array2) > 0))
    return np.sum(array1<array2) == 0

def save_node_list_to_file(node_list:list[Node], fname):
    save_data = {}
    for i, node in enumerate(node_list):
        save_data[i] = node.to_dict()
    
    with codecs.open(fname, 'w', encoding='utf8') as f: 
        json.dump(save_data, f, ensure_ascii=False, indent=4)

def dump_data(data, fname):
    with open(fname, 'wb') as f: 
        pickle.dump(data, f)

def load_data(fname) -> tuple[int, list[Node]]:
    with open(fname, 'rb') as f: 
        data = pickle.load(f)
    
    return (len(data), data)

def draw_tsp_route(figure, ax, depot_color, customer_color, n_depots, locations):
    '''
    Biểu diễn trực quan hóa đường đi TSP\n
    `figure`: \n
    `ax`: \n
    `color_list`: Mảng lưu màu hiển thị lên\n
    `locations`: tọa độ của các điểm\n
    '''

    locations = np.array(locations)
    plt.plot(locations[:n_depots, 0], locations[:n_depots, 1], depot_color)
    plt.plot(locations[:n_depots, 0], locations[:n_depots, 1], depot_color+'o')
    plt.plot(locations[n_depots:, 0], locations[n_depots:, 1], customer_color)
    plt.plot(locations[n_depots:, 0], locations[n_depots:, 1], customer_color+'o')

    plt.plot(locations[n_depots-1:n_depots+1, 0], locations[n_depots-1:n_depots+1, 1], depot_color)

    return figure, ax

def get_pole(locations):
    locations = np.array(locations)
    north = locations[np.argmax(locations, axis=0)[1]]
    east = locations[np.argmax(locations, axis=0)[0]]
    south = locations[np.argmin(locations, axis=0)[1]]
    west = locations[np.argmin(locations, axis=0)[0]]

    return north, east, west, south

def convert_demand_for_depot(demand, start_remain, end_remain, id_list, mapping_item_type):
    '''
    Demand: \n
        [[443. 138. 137. 443. 322.]\n
        [ 62. 237. 477.   0. 336.]   \n     
        [395. 211. 301. 119. 444.]      \n  
        [ 35. 305. 119. 415. 474.]\n
        [245. 441. 281. 116. 408.]\n
        [ 84. 105. 203. 346.   9.]\n
        [485. 202.  35. 273. 369.]\n
        [382. 138. 177. 157. 144.]\n
        [430. 386. 457. 251. 438.]]\n
    Start: [array([10734., 10566.])]\n
    End: [array([5229., 4096.])]\n
    Out: [\n
        {'0': array([443., 138., 137., 443., 322.]), \n
        '18': array([ 62., 237., 477.,   0., 336.]), \n
        '102': array([395., 211., 301., 119., 444.]), \n
        '106': array([ 35., 305., 119., 415., 474.]), \n
        '137': array([245., 441., 281., 116., 408.]), \n
        '148': array([ 84., 105., 203., 346.,   9.]), \n
        '185': array([485., 202.,  35., 273., 369.]), \n
        '225': array([382., 138., 177., 157., 144.]), \n
        '257': array([430., 386., 457., 251., 438.])}\n
        ]\n
    '''
    demand = np.array(demand)

    # print('Demand: {}'.format(demand))
    # print(f"Start: {start_remain}\nEnd: {end_remain}")

    res = []
    offset = np.array(start_remain) - np.array(end_remain)

    for i in range(len(start_remain)):
        res_i = {}
        for k, d in enumerate(demand):
            res_i_k = []
            if np.sum(d) == 0: continue

            if np.sum(offset[i]) == 0: break
            for j, item in enumerate(d):
                a = item
                if offset[i][int(mapping_item_type[int(j)])] >= item: 
                    demand[k][j] -= item
                    offset[i][int(mapping_item_type[int(j)])] -= item
                
                else:
                    demand[k][j] -= offset[i][int(mapping_item_type[int(j)])]
                    a = offset[i][int(mapping_item_type[int(j)])]
                    offset[i][int(mapping_item_type[int(j)])] = 0
                res_i_k.append(a)
            
            res_i[id_list[k]] = np.array(res_i_k)
        res.append(res_i)
    # print(f"Out: {np.array(res)}")
    # input('pp')
    return res
                    

def mapping_id_code(list_node):
    res = {}
    for node in list_node:
        res[node.id] = node.code
    
    return res

def mapping_code_id(list_node):
    res = {}
    for node in list_node:
        res[node.code] = node.id
    return res

def mapping_id_idx(list_node):
    res = {}
    for i in range(len(list_node)):
        res[list_node[i].id] = i

    
    return res
def change_fname(old_fname, number):
    '''
    Hàm này dùng để đổi tên file, cho phần chạy benchmark\n
    Ví dụ: market_123.json  -->  market_`number`.json
    '''
    fname = old_fname.split('.')
    fname[0] = fname[0].split('_')[0] + '_' + str(number)
    return '.'.join(fname)

from collections.abc import MutableMapping
def element_count(adict : MutableMapping) -> int:
    '''
    Đếm số phần tử trong 1 dict
    '''
    
    cnt = 0
    
    for k, v in adict.items():
        if isinstance(v, MutableMapping):
            cnt += element_count(v)
        else:
            if v is not None and v != []:
                cnt += 1
    return cnt