import numpy as np
import sys
sys.path.append("")
from src.utils.SupportClass import Node
from src.utils.uitls import *
from src.utils.get_data import *

def check_route(route, p_d_type, start_node, list_node: list[Node], s_time, correlation, move_time, item_load_time) -> tuple[list, list]:
    '''
    Kiểm tra xem khi di chuyển có time window thì thực tế trong 1 route ta có thể đi được những điểm nào, bỏ qua những điểm nào\n

    `route`: list chứa thứ tự code của các node cần di chuyển\n
    `p_d_type`: ký hiệu đỉnh là pickup hay delivery\n
    `start_node`: code của vị trí xe đang đứng \n
    `list_node`: list chứa thông tin các node trong route\n
    `s_time`: start time, đơn vị s\n
    `correlation`: quãng đường di chuyển giữa 2 node bất kỳ\n
    `move_time`:  thời gian di chuyển giữa 2 node bất kỳ\n
    `item_load_time`: mảng thời gian để load và unload 1 đơn vị hàng hóa\n
    return `arrive_list`, `ignore_list`, `c_time`\n
    '''

    item_load_time = np.array(item_load_time)

    arrive_time = np.zeros(len(route))
    leave_time = np.zeros(len(route))
    start_time = np.zeros(len(route))
    end_time = np.zeros(len(route))
    current_time = np.zeros(len(route))

    save = {'arrive_time': {}, 'leave_time': {}}

    arrive_list = [] # Mảng lưu các node có thể đến được
    ignore_list = [] # Mảng lưu các node bị bỏ qua

    c_time = s_time
    c_node = start_node # Current node

    

    # Tính toán các thông tin cần thiết
    for i in range(len(route)):

        # Lấy khối lượng cần load - unload ở điểm
        demand = np.zeros(len(item_load_time))
        if p_d_type[i] == 'D': 
            demand = list_node[i].items_array
        elif p_d_type[i] == 'P': 
            for j in range(i,len(route), 1):
                if p_d_type[j] == 'D' and str(list_node[j].id) in list_node[i].order_dict:
                    demand+= np.array(list_node[i].order_dict[str(list_node[j].id)])


        start_time[i] = list_node[i].start_time
        end_time[i] = list_node[i].end_time
        arrive_time[i] = c_time + move_time[c_node][route[i]]
        c_time += (move_time[c_node][route[i]] + np.sum(item_load_time * demand))
        leave_time[i] = c_time
        c_node = list_node[i].code
        
    # Lặp và tìm ra tuyến đường đi qua nhiều node nhất có thể
    for i in range(len(route)):
        offset = arrive_time[i] - start_time[i] 

        if offset > 0: offset = 0 
        arrive_time_i = arrive_time - offset
        leave_time_i = leave_time - offset

        save['arrive_time'][i] = arrive_time_i
        save['leave_time'][i] = leave_time_i

        arrive_list_i = []
        ignore_list_i = []
        current_time[i] = leave_time_i[-1]
        for j in range(len(route)):
            if arrive_time_i[j] >= start_time[j] and leave_time_i[j] <= end_time[j]: arrive_list_i.append(route[j])
            else:
                if j+1 == len(route): continue
                ignore_list_i.append(route[j])
                arrive_time_i[j+1:] -= (arrive_time_i[j+1] - arrive_time_i[j-1] - move_time[route[j-1]][route[j+1]]) # Node j không trong time window nên không cần đi thăm node này nữa, cập nhật lại arrive time
                leave_time_i[j+1:] -= (arrive_time_i[j+1] - arrive_time_i[j-1] - move_time[route[j-1]][route[j+1]])

        arrive_list.append(arrive_list_i)
        ignore_list.append(ignore_list_i)
    

    # Lấy ra vị trí có ignore list nhỏ nhất:
    m = len(route)
    idx = 0
    for i in range(len(ignore_list)):
        if len(ignore_list[i]) < m: 
            m = len(ignore_list[i])
            idx = i

     
    if len(arrive_list[idx]) == 0: arrive_list[idx] = None
    if len(ignore_list[idx]) == 0: ignore_list[idx] = None

    return arrive_list[idx], ignore_list[idx], current_time[idx]

def survivability(tsp_fname = 'output/vendor-depot/TSP_phase_with_Kmeans.json', tpe = 1):
    '''
    Kiểm tra xem khi thực sự chạy theo các tuyến đường đã định sẵn thì có thể giao cho được bao nhiêu điểm mà thỏa mãn time-window\n
    `tpe`:\n
    \t1: depot - customer \n
    \t2: vendor - depot \n
    '''

    config = read_config('src/config/config.yaml')

    # Load các xe vào 
    n_ve, vehicle_list = load_vehicle_from_json('input/vehicle_10.json', 0)
    
    # Khai báo các biến cần thiết
    solution = {}
    ignore = {}
    n_cus, customer_list = load_data(config['dump']['customer'])
    n_vendor, vendor_list = load_data(config['dump']['vendor'])
    n_depot, depot_list = load_data(config['dump']['depot'])
    all_list = vendor_list + depot_list + customer_list
    id_code = mapping_id_code(all_list)
    id_idx = mapping_id_idx(all_list)
    mapping_item_time = get_time_load(config['fname']['item_type'])
    correlation = json.load(open(config['fname']['correlation']))
    time_corr = json.load(open(config['fname']['time']))
    dump_file = config['output']['vendor-depot']['tsp']
    
    routes_data = json.load(open(tsp_fname, 'r'))

    for i in routes_data:
        solution[i], ignore[i] = {}, {}
        
        s_time = 0
        if tpe == 1:
            
            for k, r in enumerate(routes_data[str(i)]['route_order']):
                node_list = []
                node_type = []
                route_code = []
                if k == 0: start_node = routes_data[str(i)][str(r)]['route_code'][0]
                else: start_node = routes_data[str(i)][str(routes_data[str(i)]['route_order'][k-1])]['route_code'][-1]
                for id in routes_data[str(i)][str(r)]['route_id']:
                    node_list.append(all_list[id_idx[id]])
                route_code.extend(routes_data[str(i)][str(r)]['route_code'])
                node_type.extend(routes_data[str(i)][str(r)]['type'])

                solution[i][r], ignore[i][r], s_time = check_route(route_code, node_type, start_node, node_list, s_time, correlation, time_corr, mapping_item_time)
        
        if tpe == 2:
            for k, j in enumerate(list(routes_data[str(i)].keys())):
                node_list = []
                node_type = []
                route_code = []
                if k == 0: start_node = routes_data[str(i)][str(j)]['route_code'][0]
                else: start_node = routes_data[str(i)][str(list(routes_data[str(i)].keys())[k-1])]['route_code'][-1]

                for id in routes_data[str(i)][str(j)]['route_id']:
                    node_list.append(all_list[id_idx[id]])
                route_code.extend(routes_data[str(i)][str(j)]['route_code'])
                node_type.extend(routes_data[str(i)][str(j)]['type'])

                solution[i][j], ignore[i][j], s_time = check_route(route_code, node_type, start_node, node_list, s_time, correlation, time_corr, mapping_item_time)
            
    return solution, ignore
    


def sur_main(fname):
    config = read_config('src/config/config.yaml')
    s1, i1 = survivability(tsp_fname=config['output']['depot-customer']['tsp'], tpe = 1)
    print(f"Survivability = {round(element_count(s1) / (element_count(s1)+element_count(i1)) * 100)}%")
    # print(i1)

    s2, i2 = survivability(tsp_fname=config['output']['vendor-depot']['tsp'], tpe=2)
    print(f"Survivability = {round(element_count(s2) / (element_count(s2)+element_count(i2)) * 100)}%")

    metric = json.load(open('output/summary_TSP_with_Kmeans.json', 'r'))
    metric['depot-customer']['survivability'] = element_count(s1) / (element_count(s1)+element_count(i1))
    metric['vendor-depot']['survivability'] = element_count(s2) / (element_count(s2)+element_count(i2))

    json.dump(metric, open(fname, 'w'), indent=4)