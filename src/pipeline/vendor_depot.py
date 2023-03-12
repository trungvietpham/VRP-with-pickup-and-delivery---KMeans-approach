import sys
sys.path.append("")
from src.utils.uitls import *
from src.utils.get_data import *
from src.utils.KMeans import KMeans
from python_tsp.exact import solve_tsp_dynamic_programming
import json

def sub_cluster(vendor_id, vehicle:Vehicle, alpha):
    '''
    Chia nhỏ các node trong 1 vendor ra thành nhiều route để đảm bảo hàng hóa trong 1 route đủ để chứa trong 1 xe và số node không vượt quá threshold
    Hiện tại mới chỉ thiết lập 1 xe `vehicle`, có thể đưa lên thành 1 list các xe
    '''
    config = read_config('src/config/config.yaml', tpe='yaml')
    with open(config['fname']['correlation'], 'r') as f:
        correlation = json.load(f)
    mapping_item_type = get_items(config['fname']['item_type'])
    n_node_thr = config['other_config']['no of node threshold']

    n_vendor, vendor_list = load_data(config['dump']['vendor'])
    vendor_id_idx = mapping_id_idx(vendor_list)
    vendor = vendor_list[int(vendor_id_idx[vendor_id])]
    n_depot, depot_list = load_data(config['dump']['depot'])
    
    depot_id_idx = mapping_id_idx(depot_list)
    city_list = []
    for i in vendor.order_dict:
        if np.sum(vendor.order_dict[i]) == 0: continue
        d = depot_list[int(depot_id_idx[int(i)])]
        city_list.append(Node(d.lat, d.lng, d.id, d.code, d.name, 'CUSTOMER', items_array=vendor.order_dict[i]))

    n_sub = 1
    while True:

        child_cluster_list = []
        for i in range(n_sub):
            child_cluster_list.append(Cluster(None, None, vehicle.capacity, vehicle.coef))

        model = KMeans(n_sub)
        
        (centers, labels, it, dis, best, i_best, child_cluster_list, city_list) = model.fit(city_list, child_cluster_list, correlation, optimizer, mapping_item_type, epsilon=5*1e-6, penalty_coef=3, trade_off_coef=alpha, n_times=3)

        continue_flag = False

        for child in child_cluster_list:
            if not is_bigger_than(child.capacity, child.current_mass):
                continue_flag = True
                break

            if child.n_cities > n_node_thr:
                continue_flag = True
                break
            
            
        if continue_flag:
            n_sub+=1
            del model
            del child_cluster_list
        
        else: 
            # Thoát khỏi vòng lặp và trả về các kết quả cần thiết

            break
    
    return child_cluster_list

def vendor_tsp_phase(alpha=0.9):
    config = read_config('src/config/config.yaml')

    # Load các xe vào 
    n_ve, vehicle_list = load_vehicle_from_json('input/vehicle_10.json', 0)
    
    # Khai báo các biến cần thiết
    solution = {}
    n_vendor, vendor_list = load_data(config['dump']['vendor'])
    n_depot, depot_list = load_data(config['dump']['depot'])
    id_code = mapping_id_code(vendor_list + depot_list)
    mapping_item_type = get_items(config['fname']['item_type'])
    correlation = json.load(open(config['fname']['correlation'], 'r'))
    dump_file = config['output']['vendor-depot']['tsp']
    route_distance = []
    time_computing = []
    cost = []
    vendor_cluster = []

    for i, vendor in enumerate(vendor_list):
        res = {}

        time1 = time.time()
        child_list = sub_cluster(vendor.id, vehicle_list[i%n_ve], alpha=alpha)
        time_computing.append(time.time() - time1)

        routes = [child_list[i].cities_id for i in range(len(child_list))]
        vendor_cluster.append(child_list)
    
        for j, r in enumerate(routes):
            r = [vendor.id] + r
            l = len(r)
            corr = np.zeros((l, l))
            idx_id = []
            
            for a in range(l):
                idx_id.append(r[a])
                for b in range(l):
                    # print(f"a: {a}, b: {b}, ra: {r[a]}, id_c: {id_code[r[a]]}")
                    # input('PP')
                    corr[a][b] = correlation[id_code[r[a]]][id_code[r[b]]]
            
            time1 = time.time()
            perm, dis = solve_tsp_dynamic_programming(corr)
            time_computing.append(time.time() - time1)

            route_distance.append(dis)  
            # print(f"Coef = {vehicle_list[i%n_ve].coef}, dis = {dis}, goods percentage = {(child_list[j].current_mass)}")
            # input('p')
            cost.append(vehicle_list[i%n_ve].coef * dis * (np.sum(child_list[j].current_mass) / np.sum(vehicle_list[i%n_ve].capacity)))
            
            p_d_type = ['P']
            for _ in range(len(perm)-1): p_d_type.append('D')
            res[j] = {'route_id': [idx_id[i] for i in perm], 'route_code': [id_code[idx_id[i]] for i in perm], 'type': p_d_type}
            res[j]['length'] = f"{str(round(dis/1000, 0))} km"
        solution[i] = res

    # Dump vào file
    folder = os.path.dirname(dump_file)
    if not os.path.exists(folder): 
            os.makedirs(folder)
    json.dump(solution, open(dump_file, 'w'), indent=4)

    route_distance = np.array(route_distance) / 1000
    summary_for_compare = {'vendor-depot' :{'Distance': np.sum(np.array(route_distance)), 'Time': round(np.sum(np.array(time_computing))*1000.0, 0), 'Cost':round(np.sum(cost) / 1000)}}
    a = json.load(open('output/summary_TSP_with_Kmeans.json', 'r'))
    a.update(summary_for_compare)
    json.dump(a, open('output/summary_TSP_with_Kmeans.json', 'w'), indent=4)

    # Dump dữ  liệu các cluster vào file
    dump_data(vendor_cluster, 'dump/vendor_cluster.pkl')

    summary = []
    details = []

    summary.append('\tSummary: ')
    summary.append('\t\tTotal route length = {} (km)'.format(np.sum(np.array(route_distance))))
    summary.append('\t\tTotal time computing TSP = {} ms'.format(round(np.sum(np.array(time_computing))*1000.0, 0)))
    summary.append('\t\tTotal cost = {}'.format(np.sum(cost)))

    details.append('\n'.join(summary))
    details.append('\tDetails: ')

    cnt = 0

    for i,v in enumerate(vendor_list):
        details.append('\t\t\tVehicle ID: {}'.format(vehicle_list[i%n_ve].id))
        details.append('\t\t\tNo. of routes: {}'.format(len(solution[i])))
        total_cost = np.sum(cost[cnt:cnt+len(solution[i])])
        total_distance = np.sum(route_distance[cnt:cnt+len(solution[i])])
        details.append('\t\t\tTotal length: {} (km)'.format(total_distance))
        details.append('\t\t\tTotal cost: {}'.format(total_cost))
        details.append('\t\t\tAll route details: ')

        for j in solution[i]:
            details.append('\t\t\t\tRoute: {}'.format(' -> '.join(solution[i][j]['route_code'])))
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
    return ('\n'.join(summary), '\n'.join(details), n_vendor, total_time, total_route_length, total_cost)

if __name__ == '__main__':
    vendor_tsp_phase() 

