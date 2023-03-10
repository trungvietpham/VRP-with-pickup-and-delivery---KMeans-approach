import time
import numpy as np
import json
from python_tsp.exact import solve_tsp_dynamic_programming

import sys
import os
sys.path.append("")
from src.utils.uitls import *
from src.utils.get_data import *

def get_best_point(point, depot_list, strategy = 'nearest', point_demand = None, depot_capacity = None):
    '''
    Trả về chỉ số của `depot_list` phù hợp nhất so với `point` theo chiến lược `strategy`\n
    `point` = [lat, long]: là tọa độ của tâm cụm\n
    `depot_list` = [[lat, long]]*n_depots\n
    `strategy`: chiến lược tìm điểm tốt nhất, nhận các giá trị: 'nearest', 'best fit', 'ensemble'\n
    \t'nearest': Tìm điểm gần `point` nhất\n
    \t'best fit`: Tìm điểm có capacity phù hợp nhất\n
    \t`ensemble`: Kết hợp cả 2 cái trên\n
    `point_demand`: Demand của `point`, sử dụng khi `strategy` = 'nearest'\n
    `depot_capacity`: capacity của từng depot, sử dụng khi `strategy` = 'nearest'\n

    Tính khoảng cách giữa point đối với từng điểm depot và 
    sắp xếp thành 1 mảng tăng dần về khoảng cách
    '''
    strategy_list = ['nearest', 'best fit', 'ensemble']
    assert strategy in strategy_list, 'strategy must be in {}'.format(strategy_list)

    depot_list = np.array(depot_list)
    dis = np.zeros(len(depot_list))
    for i in range(len(depot_list)):
        dis[i] = distance(point, depot_list[i])
    argsort = np.argsort(dis)
    if strategy == 'nearest':
        
        return argsort[0]
    
    elif strategy == 'best fit':
        found_flag = False
        for i in argsort:
            if is_bigger_than(depot_capacity[i], point_demand): 
                found_flag = True
                return i
        
        if found_flag == False:
            print('Best fit not found, change strategy to nearest')
            return get_best_point(point, depot_list, strategy='nearest')



def TSP_phase(vehicle_fname, tpe = 'depot-customer'):
    tpe_list = ['depot-customer', 'vendor-depot']
    assert tpe in tpe_list, 'tpe must be in {}, found {}'.format(tpe_list, tpe)
    
    # Đọc config:
    config_fname = 'src/config/config.yaml'
    config = read_config(config_fname, tpe='yaml')

    correlation_fname = config['fname']['correlation']
    item_type_fname = config['fname']['item_type']
    if tpe == 'depot-customer':
        depot_fname = config['fname']['depot']
        customer_fname = config['fname']['customer']
    
    elif tpe == 'vendor-depot':
        depot_fname = config['fname']['vendor']
        customer_fname = config['fname']['depot']
    vendor_fname = config['fname']['vendor']
    #Load correlation 
    n_items = config['other_config']['no of items']
    n_items_type = config['other_config']['no of items type']
    correlation = json.load(open(correlation_fname, 'r'))
    cluster_data = json.load(open(config['output'][tpe]['pre_tsp'], 'r'))
    (_, vehicle_list) = load_vehicle_from_json(vehicle_fname, n_items)
    mapping_item_type = get_items(item_type_fname)
    dump_file = config['output'][tpe]['tsp']

    summary = []
    details = []
    details.append('\nDescription: Use TSP algorithm for each sub-cluster\n')

    # Lấy thông tin của các depot:
    if tpe == 'depot-customer':
        n_d, depot_list = load_node_from_json(depot_fname, 'depot', n_items_type)
        n_customer, customer_list = load_data(config['dump']['customer'])
    
    elif tpe == 'vendor-depot':
        n_d, depot_list = load_node_from_json(depot_fname, 'vendor', n_items_type)
        n_customer, customer_list = load_data(config['dump']['depot'])
        n_items_type = n_items
    
    n_v, vendor_list = load_node_from_json(vendor_fname, 'vendor', n_items_type)

    #Lấy ra toàn bộ tọa độ của các depot và lưu vào list
    depot_locations = []
    for depot in depot_list:
        depot_locations.append(depot.get_location())
    del depot

    customer_locations = []
    for customer in customer_list:
        customer_locations.append(customer.get_location())
    del customer

    # Kiểm tra xem capacity của các depot có lớn hơn demand của các customer (kiểm tra tính khả thi của bài toán)
    depot_capa = []
    customer_demand = []
    for depot in depot_list:
        if tpe == 'depot-customer':
            depot_capa.append(depot.items_type_array)
        elif tpe == 'vendor-depot':
            depot_capa.append(depot.items_array)

    for customer in customer_list:
        if tpe == 'depot-customer':
            customer_demand.append(convert_demand(customer.items_array, mapping_item_type))
        elif tpe == 'vendor-depot':
            customer_demand.append(customer.items_array)

    if not is_bigger_than(np.sum(depot_capa, axis=0), np.sum(customer_demand, axis=0)):
        print('Bài toán không khả thi')
        print('Total depot capacity: {}'.format(np.sum(depot_capa, axis=0)))
        print('Total demand: {}'.format(np.sum(customer_demand, axis=0)))
        return -1, 0, n_d, 0, 0, 0
    del depot_capa
    del customer_demand

    # Mapping code của node với tọa độ của node: 
    mapping_code_location = {}
    for depot in depot_list:
        mapping_code_location[depot.code] = depot.get_location()
    for customer in customer_list:
        mapping_code_location[customer.code] = customer.get_location()
    
    # Mapping code của depot với chỉ số lưu trong list:
    mapping_code_index = {}
    for i, depot in enumerate(depot_list):
        mapping_code_index[depot.id] = i
    
    id_code_map = mapping_id_code(customer_list)
    id_code_map.update(mapping_id_code(depot_list))


    # Dict lưu lại các thông tin để lưu trữ
    save_data = {}

    # Một số biến lưu trữ tổng độ dài TSP, thời gian tính toán
    route_distance = []
    adding_route_distance = []
    meta_route_demand = [] # Lưu demand của route (đã convert về demand theo item type)
    meta_origin_route_demand = [] # Lưu demand của route chưa convert 
    time_computing = []
    v_coef_list = []
    goods_percentage = [] # Mảng lưu tỉ lệ tổng hàng hóa yêu cầu / sức chứa của xe
    route_list = []

    

    #Lặp qua các cụm con và tsp 
    for cluster_parent_key in cluster_data:

        start_list = []
        end_list = []

        center_parent = cluster_data[cluster_parent_key]['center']
        n_cluster_child = len(cluster_data[cluster_parent_key]["child_cluster_list"])
        cluster_info = {}
        # cluster_info['cluster_id'] = cluster_parent_key
        # cluster_info['center'] = cluster_data[cluster_parent_key]['center']

        for cluster_child_key in cluster_data[cluster_parent_key]["child_cluster_list"]:
            
            cluster_child = cluster_data[cluster_parent_key]["child_cluster_list"][cluster_child_key]
            n_node_child = len(cluster_child["node_list"])

            # Tính tổng demand của các node trong cụm con:
            route_demand = np.zeros(n_items) # Lưu tổng demand trong cụm con
            demand_list = []
            for id in cluster_child['node_list']:
                node_demand = []
                for k in range(n_items):
                    node_demand.append(cluster_child['node_list'][id]['demand']['Item {}'.format(k)])
                route_demand += np.array(node_demand)
                demand_list.append(node_demand)
            
            meta_origin_route_demand.append(route_demand.copy())
            if tpe == 'depot-customer': route_demand = convert_demand(route_demand, mapping_item_type)

            meta_route_demand.append(route_demand.copy())

            # Tìm các depot để thỏa mãn có thể cung cấp đủ cho demand của các customer
            continue_flag = True
            route_depot_list = []
            point1 = [cluster_child["center"]['lat'], cluster_child["center"]['long']]
            list_point_2 = depot_locations.copy()
            index_list = [k for k in range(len(depot_locations))]
            total_depot_capa = np.zeros(n_items_type) # Lưu tổng capacity của các depot được gắn với các customer trong route
            depot_capacity_list = np.zeros(n_items_type) 
            remain_list = []
            for depot in depot_list:
                remain_list.append(depot.remain_capacity)
            
            # Lưu lại thông tin của depot remain trước và sau khi được phân vào route
            start_remain = []
            end_remain = []

            # Loại bỏ những depot có remain = 0
            i = 0
            while True:
                if i>=len(index_list): break
                if np.sum(np.array(remain_list[i]) != 0) == 0: 
                    del list_point_2[i]
                    del remain_list[i]
                    del index_list[i]
                else: i+=1

            while continue_flag:
                continue_flag = False
                index = get_best_point(point1, list_point_2, strategy='best fit', point_demand=route_demand, depot_capacity=remain_list)
                start_remain.append(depot_list[index_list[index]].remain_capacity.copy())
                total_depot_capa += np.array(start_remain[-1].copy())
                depot_capacity_list += np.array(depot_list[index_list[index]].remain_capacity.copy())
                remain_list[index] = depot_list[index_list[index]].remain_capacity.copy()
                
                # Update lại remain_capacity của depot:
                old_remain = depot_list[index_list[index]].remain_capacity
                new_remain = old_remain - route_demand
                new_remain = (new_remain>=0) * new_remain # Nếu âm thì đưa về = 0
                depot_list[index_list[index]].update_remain_capacity(old_remain - new_remain)
                end_remain.append(depot_list[index_list[index]].remain_capacity.copy())

                route_depot_list.append(depot_list[index_list[index]].id)

                # TODO
                if not is_bigger_than(depot_capacity_list, route_demand):
                    # Nếu sai thì cập nhật lại các biến để tiếp tục vòng lặp
                    continue_flag = True
                    point1 = depot_locations[index_list[index]]
                    del list_point_2[index]
                    del index_list[index]
                    route_demand -= (old_remain - new_remain)

            # TODO: Thêm module để phân các đơn hàng vào từng depot để shuffle cho vendor gửi hàng
            # Gán vào items-array của depot
            depot_demand = convert_demand_for_depot(demand_list, start_remain, end_remain, list(cluster_child["node_list"].keys()), mapping_item_type)
            for idx, d in enumerate(route_depot_list):
                
                if depot_list[int(mapping_code_index[route_depot_list[idx]])].items_array is None: 
                    depot_list[int(mapping_code_index[route_depot_list[idx]])].items_array = np.zeros(int(config['other_config']["no of items"]))

                depot_list[int(mapping_code_index[route_depot_list[idx]])].add_order(depot_demand[idx])

            #Mapping từ số trong khoảng (0, n_node_child) về id node 
            mapping_code = {}
            mapping_id = {}
            reverse = {}
            cnt = -1
            for i in range(len(route_depot_list)):
                mapping_code[i] = id_code_map[route_depot_list[i]]
                mapping_id[i] = route_depot_list[i]
                reverse[route_depot_list[i]] = i
                cnt+=1
            end_depot_flag = cnt
            cnt+=1
            # Tiếp tục mapping:
            for id in cluster_child['node_list']:
                mapping_code[cnt] = id_code_map[int(id)]
                mapping_id[cnt] = int(id)
                reverse[mapping_code[cnt]] = cnt
                cnt+=1

            # Lưu lại các biến cho phần tính toán cost
            goods_percentage.append(np.sum(route_demand)/np.sum(vehicle_list[int(cluster_parent_key)].capacity))
            v_coef_list.append(vehicle_list[int(cluster_parent_key)].coef)

            # Tạo 1 mảng 2 chiều là distance giữa 2 node bất kỳ trong đây, 
            distance_matrix = np.zeros((n_node_child+1, n_node_child+1))
            # print('Mapping list: {}'.format(mapping))
            for i in range(n_node_child+1):
                for j in range(n_node_child+1):
                    if mapping_code[j+end_depot_flag] not in correlation[mapping_code[i+end_depot_flag]]: correlation[mapping_code[i+end_depot_flag]][mapping_code[j+end_depot_flag]] = 0.0 
                    distance_matrix[i][j] = correlation[mapping_code[i+end_depot_flag]][mapping_code[j+end_depot_flag]]
            
            #TSP
            time1 = time.time()
            permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
            time2 = time.time()

            reverse_permutation = [mapping_code[i] for i in range(end_depot_flag - 1)]
            route_id = [mapping_id[i] for i in range(end_depot_flag - 1)]

            p_d_type = ['P' for i in range(end_depot_flag+1)] # pickup delivery type, in ['D', 'P']
            # print(f"end flag: {end_depot_flag}")
            # print(f"pd: {p_d_type}, route id: {route_id}")
            # input('PD')
            for i in range(n_node_child+1):
                reverse_permutation.append(mapping_code[int(permutation[i]) + end_depot_flag])
                route_id.append(mapping_id[int(permutation[i]) + end_depot_flag])

            for i in range(n_node_child):
                p_d_type.append('D')

            dist_res = 0.0
            for i in range(1, len(reverse_permutation)):
                dist_res += float(correlation[reverse_permutation[i-1]][reverse_permutation[i]]/1000)

            dist_res = round(dist_res, 0)
            route_list.append(' -> '.join(reverse_permutation))
            cluster_info[cluster_child_key] = {}
            cluster_info[cluster_child_key]['route_id'] = route_id
            cluster_info[cluster_child_key]['route_code']= reverse_permutation
            cluster_info[cluster_child_key]['type'] = p_d_type
            cluster_info[cluster_child_key]['length'] = str(round(dist_res, 0)) + ' km'

            #In ra thông tin tiến trình
            print('\r\tDone {}/{}. Output in {}'.format(int(cluster_child_key)+1, len(cluster_data[cluster_parent_key]["child_cluster_list"]), dump_file))

            # Lưu lại các thông tin về khoảng cách, thời gian tính toán
            route_distance.append(dist_res)
            time_computing.append(time2-time1)
            start_list.append(reverse_permutation[0])
            end_list.append(reverse_permutation[-1])

            #Lưu dữ liệu
            json.dump(save_data, open(dump_file, 'w'), indent=4)

            #Xóa bộ nhớ
            del cluster_child, mapping_code, reverse
        
        meta_corr = np.zeros((len(start_list), len(start_list)))
        for i in range(len(start_list)):
            for j in range(len(start_list)):
                meta_corr[i][j] = correlation[end_list[i]][start_list[j]]
        
        perm, dis = solve_tsp_dynamic_programming(meta_corr)
        adding_route_distance.append(dis/1000)

        cluster_info['route_order'] = perm
        save_data[cluster_parent_key] = cluster_info
        


    # Tính chi phí xe di chuyển trong 1 route, cost = (hệ số xe) * (tổng khoảng cách xe di chuyển) * (tỉ lệ hàng hóa trên xe)
    cost = np.round(np.array(v_coef_list) * (np.array(route_distance)) * np.array(goods_percentage), decimals=1)

    # shuffle các order về cho vendor
    vendor_code_id = mapping_code_id(vendor_list)
    vendor_id_idx = mapping_id_idx(vendor_list)
    for i in range(len(depot_list)):
        # print(f"O dict: {depot_list[i].order_dict}")
        # input('P')
        for c_id in depot_list[i].order_dict:
            # print(f"Code id: {vendor_code_id[customer_list[int(c_id)].seller]}")
            # print(f"Seller: " )
            vendor_list[vendor_id_idx[int(vendor_code_id[customer_list[int(c_id)].seller])]]._add_order(depot_list[i].id, depot_list[i].order_dict[c_id])

    #Dump ra file 'output/TSP_phase.json'
    folder = os.path.dirname(dump_file)
    if not os.path.exists(folder): 
            os.makedirs(folder)
    json.dump(save_data, open(dump_file, 'w'), indent=4)

    # Dump ra file pkl để lưu dữ liệu:
    depot_dump_fname = config['dump']['depot']
    if not os.path.exists(os.path.dirname(depot_dump_fname)): os.makedirs(os.path.dirname(depot_dump_fname))
    dump_data(depot_list, depot_dump_fname)

    dump_data(vendor_list, config['dump']['vendor'])

    
    # Trực quan hóa dữ liệu
    n,e,w,s = get_pole(depot_locations + customer_locations)
    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(len(route_list)):
        plt.xlim(w[0], e[0])
        plt.ylim(s[1], n[1])
        route = route_list[i].split(' -> ')
        node_tpe = [r[0] for r in route]
        n_depots = np.sum(np.array(node_tpe) == 'D')
        locations = []
        for code in route:
            locations.append(mapping_code_location[code])
        figure,ax = draw_tsp_route(figure, ax, 'r', 'b', n_depots, locations)

        figure.canvas.draw()
        figure.canvas.flush_events()

        print(f"Route: {route_list[i]}, length: {route_distance[i]} km")
        if input('Press to continue (0 to exit)...') == '0': break
        plt.clf()
    




    summary_for_compare = {'depot-customer' :{'Distance': np.sum(np.array(route_distance)) + np.sum(adding_route_distance), 'Time': round(np.sum(np.array(time_computing))*1000.0, 0), 'Cost':round(np.sum(cost))}}
    json.dump(summary_for_compare, open('output/summary_TSP_with_Kmeans.json', 'w'), indent=4)

    summary.append('\tSummary: ')
    summary.append('\t\tTotal route length = {} (km)'.format(np.sum(np.array(route_distance))))
    summary.append('\t\tTotal time computing TSP = {} ms'.format(round(np.sum(np.array(time_computing))*1000.0, 0)))
    summary.append('\t\tTotal cost = {}'.format(np.sum(cost)))

    details.append('\n'.join(summary))
    details.append('\tDetails: ')

    cnt = 0

    for cluster_parent_key in cluster_data:
        details.append('\t\tCluster parent: {}'. format(cluster_parent_key))
        details.append('\t\t\tVehicle ID: {}'.format(vehicle_list[int(cluster_parent_key)].id))
        details.append('\t\t\tNo. of routes: {}'.format(len(cluster_data[cluster_parent_key]["child_cluster_list"])))
        total_cost = np.sum(cost[cnt:cnt+len(cluster_data[cluster_parent_key]["child_cluster_list"])])
        total_distance = np.sum(route_distance[cnt:cnt+len(cluster_data[cluster_parent_key]["child_cluster_list"])])
        details.append('\t\t\tTotal length: {} (km)'.format(total_distance))
        details.append('\t\t\tTotal cost: {}'.format(total_cost))
        details.append('\t\t\tAll route details: ')

        for cluster_child_key in cluster_data[cluster_parent_key]["child_cluster_list"]:
            details.append('\t\t\t\tRoute: {}'.format(route_list[cnt]))
            details.append('\t\t\t\tTSP route length: {} (km)'.format(route_distance[cnt]))
            details.append('\t\t\t\tRoute cost: {}'.format(cost[cnt]))
            details.append('\t\t\t\tTime computing TSP: {}\n'.format(round(time_computing[cnt]*1000.0, 0)))
            cnt+=1
    
    details.append('\n\n')
    summary.append('\n\n')

    print(''.join(summary))
    total_time = round(np.sum(np.array(time_computing))*1000.0, 0)
    total_route_length = np.sum(np.array(route_distance))
    total_cost = np.sum(cost)
    return ('\n'.join(summary), '\n'.join(details), n_d, total_time, total_route_length, total_cost)


if __name__ == "__main__":
    TSP_phase('input/vehicle_20.json')


