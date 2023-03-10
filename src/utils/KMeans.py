# Kiểm tra lại hàm chuẩn hóa

import os
import sys
sys.path.append("")
import numpy as np
from src.utils.uitls import *
from src.utils.SupportClass import *
class KMeans:
    def __init__(self, k) -> None:
        self.k = k

    def init_center(self, locations):
        '''
        Params: \n
        city_array: list class City\n
        Return value: array(k, n_dims) lưu thông tin tọa độ init của k cụm\n
        
        Cách khởi tạo: Lấy ngẫu nhiên k điểm trong số các điểm trong locations để làm tâm cụm khởi đầu \n
        '''
        self.n_cities = len(locations)
        
        index_list = np.random.choice(int(len(locations)), self.k, replace=False)
        centroid = []
        for i in index_list:
            centroid.append(locations[i])
        
        self.init_centroid = np.array(centroid)
        return np.array(centroid)

    def assign_labels(self, optimizer, locations, centroid, mapping_item_type, shuffle = True):
        '''
        Gán mỗi điểm vào cụm gần nhất theo hàm `optimizer`\n
        Danh sách tham số: \n
            `optimizer`: Hàm tối ưu\n
            `locations`: Thông tin về tọa độ các node\n
            `centroid`: Thông tin về tọa độ các tâm\n
            `shuffle`: Cờ để trộn data\n
            `convert_flag`: Cờ convert cho hàm optimizer\n
        Return value: array(n_nodes,) lưu id của cụm mà node được gán vào\n
        '''
        # Cast params thành array
        locations = np.array(locations)
        centroid = np.array(centroid)

        #Shuffle các locations và demands
        index_list = np.arange(locations.shape[0])
        if shuffle:
            np.random.shuffle(index_list)
        shuffle_locations = [locations[i] for i in index_list]
        shuffle_locations = np.array(shuffle_locations)
        nodes_demands_shuffled = np.array([self.demand_list[i] for i in index_list])
        
        reverse_index = np.zeros(index_list.shape)
        for i, val in enumerate(index_list): reverse_index[val] = i

        # Chuẩn hóa dữ liệu trước khi đưa vào hàm optimizer:
        clusters_capacity_normed = self.clusters_capacity
        nodes_demand_normed = nodes_demands_shuffled

        res_matrix = np.array(optimizer(clusters_capacity_normed, nodes_demand_normed, shuffle_locations, centroid,
                                penalty_coef = self.penalty_coef, 
                                scale_coef=self.scale_coef_norm, 
                                trade_off_coef=self.trade_off_coef,
                                item_type_mapping = mapping_item_type
                                ))

        res = np.argmin(res_matrix, axis=1)

        # Reshuffle lại theo đúng thứ tự ban đầu
        reshuffle_res = [res[int(i)] for i in reverse_index]

        return np.array(reshuffle_res)



    def update_centers(self, locations, labels):
        '''
        Cập nhật lại tọa độ tâm cụm\n
        Return value: array(k, n_dims) lưu thông tin tọa độ sau khi update của k cụm\n
        '''
        locations = np.array(locations)
        labels = np.array(labels)
        centers = np.zeros((self.k, locations.shape[1]))
        for i in range(self.k):
            locations_i = locations[labels == i, :]
            if len(locations_i) != 0:
                centers[i,:] = np.mean(locations_i, axis=0)
            else: centers[i, :] = locations[np.random.choice(int(len(locations)), 1, replace=False)[0]]
        
        return centers

    def has_converged(self, centers, new_centers, epsilon = 1e-8):

        # Cast data
        centers = np.array(centers)
        new_centers = np.array(new_centers)
        
        # Check data
        assert centers.shape == new_centers.shape, '2 input array must have same shape'

        
        diff = 0
        for i in range(self.k):
            diff += distance(centers[i, :], new_centers[i, :])/self.distance_coef
        
        diff /= self.k
        # print('Diff = {}'.format(diff))
        return diff <= epsilon

    def data_normalize(self):
        '''
        Chuẩn hóa dữ liệu về khoảng (0,1) để tiện cho việc tính toán\n
        return: (distance_coef, mass_coef, scale_coef_norm) với distance_coef là hệ số khoảng cách, là max các khoảng cách giữa các nodes, mass_coef là hệ số chuẩn hóa của khối lượng hàng hóa ở các node
        '''

        distance_coef = 0
        for i in self.correaltion:
            for j in self.correaltion[i]:
                if distance_coef < self.correaltion[i][j]: distance_coef = self.correaltion[i][j] 

        # Tính mass_coef
        mass_coef = np.max(self.demand_list)

        # Tính scale_coef_norm
        scale_coef_norm = np.max(self.scale_coef)

        # Lưu vào thành thuộc tính của lớp: 
        self.distance_coef = distance_coef # Hệ số chuẩn hóa của khoảng cách
        self.mass_coef = mass_coef # Hệ số chuẩn hóa của demand
        self.scale_coef_norm = scale_coef_norm # Hệ số chuẩn hóa của scale_coef

        return (distance_coef, mass_coef, scale_coef_norm)


    def fit(self, node_list: list[Node], cluster_list: list[Cluster], correlation, optimizer: optimizer, mapping_item_type, epsilon: float = 1e-8, penalty_coef: float = 2.0, trade_off_coef = 0.5, shuffle=True, n_times = 5):
        '''
        Hàm học thuật toán KMeans, danh sách tham số: \n
        `node_list`: danh sách các node trong không gian phân cụm (sử dụng cấu trúc Node)\n
        `cluster_list`: danh sách các cluster (sử dụng cấu trúc Cluster)\n
        `correlation`: mảng lưu khoảng cách giữa 2 điểm bất kỳ trong tập hợp {depots + customers}
        `optimizer`: Hàm tối ưu được sử dụng\n
        `mapping_item_type`: Mảng lưu thông tin về item_type của từng item\n
        `epsilon`: Ngưỡng xảy ra điều kiện dừng\n
        `penalty_coef`: Hệ số phạt trong trường hợp vượt quá dung tích được gán cho cluster\n
        `trade_off_coef`: Hệ số alpha giữa khoảng cách và demand trong hàm optimizer, thỏa mãn alpha trong khoảng (0,1) \n
        `shuffle`: Cờ để xem có trộn bộ dữ liệu Node lại trước khi cho qua bước học tiếp theo không\n
        `n_times`: Số lần lặp để chạy lại thuật toán\n

        '''
        # Lưu các thông tin cần thiết ra mảng: 
        id_list = []
        demand_list = []
        locations = []
        clusters_capacity = []
        scale_coef_list = []

        for node in node_list:
            id_list.append(node.id)
            demand_list.append(node.items_array)
            locations.append(node.get_location())

        for cluster in cluster_list:
            clusters_capacity.append(cluster.capacity)
            scale_coef_list.append(cluster.scale_coef)

        # Cast data sang array: 
        id_list = np.array(id_list)
        demand_list = np.array(demand_list)
        locations = np.array(locations)
        clusters_capacity = np.array(clusters_capacity)
        scale_coef_list = np.array(scale_coef_list)
        mapping_item_type = np.array(mapping_item_type)

        # Tạo các thuộc tính của lớp:
        self.id_list = id_list
        self.demand_list = demand_list
        self.locations = locations
        self.clusters_capacity = clusters_capacity
        self.scale_coef = scale_coef_list
        if correlation is not None:
            self.correaltion = correlation
        self.penalty_coef = penalty_coef
        self.trade_off_coef = trade_off_coef
        
        # Lấy các giá trị để chuẩn hóa: 
        self.data_normalize()

        # Khởi tạo các biến lưu lại các thông tin trung gian: 
        # Số lần chạy lại để lấy ra kết quả tối ưu
        meta_it = []
        meta_all_centroids = []
        meta_all_labels = []
        meta_total_dis = []
        meta_best = []
        meta_i_best = []

        
        # Cho chạy thuật toán n_times lần và lấy ra kết quả tốt nhất để return
        for j in range(n_times):
            # Khởi tạo tâm cụm
            centroid = self.init_center(locations)
            labels = np.array([-1] * self.n_cities)

            # Khởi tạo các biến cho mỗi bước lặp:
            it = 0 # Số bước lặp  
            all_centroids = [centroid] # Thông tin về tọa độ tâm cụm mỗi bước lặp 
            all_labels = [labels] # Thông tin về nhãn của các node mỗi bước lặp 
            total_dis = [] # Thông tin về tổng khoảng cách từ tâm cụm tới các node trong cụm sau mỗi bước lặp
            check_converged = False

            # Lặp thuật toán tới khi đạt điều kiện dừng
            while not check_converged:
                
                # Cần chuẩn hóa dữ liệu trước khi đưa vào assign_labels (pass vào bằng các thuộc tính của lớp)
                it+=1
                all_labels.append(np.array(self.assign_labels(optimizer, locations, all_centroids[-1], mapping_item_type, shuffle=shuffle)))
                all_centroids.append(self.update_centers(locations, all_labels[-1]))

                # Tính tổng khoảng cách từ tâm cụm tới các node trong cụm:
                total_dis.append(total_distance(all_centroids[-1], locations, all_labels[-1]))

                check_converged = self.has_converged(all_centroids[-2], all_centroids[-1], epsilon=epsilon)

            meta_it.append(it)
            meta_all_centroids.append(all_centroids)
            meta_all_labels.append(all_labels)
            meta_total_dis.append([total_dis[0]] + total_dis)
            meta_best.append(np.min(meta_total_dis[-1]))
            meta_i_best.append(len(meta_total_dis[-1]) - 1 - np.argmin(np.flip(meta_total_dis[-1])))

        # Lựa chọn lượt có giá trị total_dis nhỏ nhất và trả về
        best = np.min(meta_best)
        i_best = np.argmin(meta_best)

        # Lưu các thông tin vào các thuộc tính của cụm
        cluster_list, node_list = save_data_to_cluster_customer(cluster_list, meta_all_centroids[i_best][meta_i_best[i_best]], meta_all_labels[i_best][meta_i_best[i_best]], node_list, mapping_item_type)

        return (meta_all_centroids[i_best], meta_all_labels[i_best], meta_it[i_best], meta_total_dis[i_best], best, meta_i_best[i_best], cluster_list, node_list)

    def visualize(self, locations, center, labels, K):
        kmeans_display(locations, center, labels, K)