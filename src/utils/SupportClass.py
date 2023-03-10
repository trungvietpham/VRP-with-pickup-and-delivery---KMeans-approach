import numpy as np
class Node:
    def __init__(self, lat: float, lng:float, id: int, code: str, name:str, tpe: str, items_array = None, items_type_array = None, cluster_id: int = None, remain_capacity = None, seller = None, start_time = None, end_time = None) -> None:
        '''
        lat, lng: tọa độ của điểm\n
        id:\n
        code: \n
        name: \n
        tpe: Loại điểm (customer, depot, vendor)\n
        items_array: Mảng lưu yêu cầu (nếu node là customer hoặc depot, dùng khi phân cụm giữa vendor - depot) hoặc lưu số lượng sẵn có theo item (nếu node là vendor)\n
        items_type_array: Mảng lưu sức chứa đối với từng loại mặt hàng (dùng khi node là depot) \n
        cluster_id: id của cụm chứa điểm này\n
        remain_capacity: Mảng lưu sức chứa còn lại của Node đối với từng `items_type_array` (sử dụng khi node là depot)\n
        order_to: Code của vendor mà customer order hàng\n
        '''
        self.lat = lat
        self.lng = lng
        self.id = id
        self.code = code
        self.name = name
        self.type = tpe
        if self.type == 'CUSTOMER' or self.type == 'VENDOR':
            self.items_array = np.array(items_array)
        else: self.items_array = None
        if self.type == 'DEPOT': 
            self.items_type_array = np.array(items_type_array)
        else: self.items_type_array = None
        self.cluster_id = cluster_id

        if remain_capacity is None: 
            if tpe == 'CUSTOMER': 
                self.remain_capacity = np.array(self.items_array)
            elif tpe == 'DEPOT': 
                self.remain_capacity = np.array(self.items_type_array)
            elif tpe == 'VENDOR': 
                self.remain_capacity = np.array(self.items_array)
        else: self.remain_capacity = np.array(remain_capacity)

        self.seller = seller # Lưu code người bán (customer order từ vendor nào)
        # self.demand_dict = {} # Lưu thông tin cho depot, xem demand bao nhiêu đối với vendor nào
        self.order_dict = {} # Lưu thông tin depot (vendor) cần cung cấp cho customer (depot) nào, khối lượng bao nhiêu

        self.start_time = start_time
        self.end_time = end_time
        

    def get_location(self):
        return [self.lat, self.lng]
    
    def update_remain_capacity(self, demand, type = 'm'):
        '''
        Cập nhật remain_capacity\n
        Tham số: \n
        demand: mảng lưu khối lượng cần cập nhật\n
        type: loại cập nhật, nhận 2 giá trị là 'a': add, 'm': minus\n
        '''
        if type == 'a': 
            self.remain_capacity += np.array(demand)
        elif type == 'm':
            self.remain_capacity -= np.array(demand)
    
    def _add_order(self, customer_code, order):
        '''
        customer_code: \n
        order: demand của customer
        '''
        if customer_code in self.order_dict:
            self.order_dict[customer_code] += order
        else: self.order_dict[customer_code] = order

    def _remove_order(self, code):
        if code in self.order_dict:
            del self.order_dict[code]
    
    def add_order(self, o_dict):
        for key in o_dict:
            self._add_order(key, o_dict[key])


    # def _add_demand(self, vendor_code, order):
    #     if vendor_code in self.demand_dict:
    #         self.demand_dict[vendor_code] += order
    #     else: self.demand_dict[vendor_code] = order
    
    # def _remove_demand(self, code):
    #     if code in self.demand_dict:
    #         del self.order_dict[code]
    
    # def add_demand(self, d_dict):
    #     for key in d_dict:
    #         self._add_demand(key, d_dict[key])


    def print(self, id_flag = False, location_flag = False, demand_flag = False, cluster_id_flag = False, remain_capa_flag = False, header = ''):
        if id_flag: print('{}Id: {}'.format(header, self.id))
        if location_flag: print('{}Location: ({}, {})'.format(header, self.lat, self.lng))
        if demand_flag: 
            if self.type == 'DEPOT': print('{}Demand: {}'.format(header, self.items_type_array))
            elif self.type == 'CUSTOMER': print('{}Demand: {}'.format(header, self.items_array))
            elif self.type == 'VENDOR': print('{}Demand: {}'.format(header, self.items_type_array))
        if remain_capa_flag: print('{}Remain capacity: {}'.format(header, self.remain_capacity))
        if cluster_id_flag: print('{}Cluster ID: {}'.format(header, self.cluster_id))

    def to_dict(self):
        return self.__dict__
        
class Vehicle:
    '''
    Lớp chứa thông tin về 1 xe, gồm: id, mảng sức chứa tối đa đối với mỗi mặt hàng
    '''
    def __init__(self, id, capacity, v_type, coef):
        '''
        id: định danh của xe\n
        capacity: Tải trọng của xe với từng loại mặt hàng (item type, là phân cấp mức cao hơn)\n
        item_capacity: Tải trọng của xe đối với từng mặt hàng (mỗi mặt hàng cụ thể ứng với 1 slot ở đây) (xem xét bỏ qua)\n
        v_type: loại xe\n
        coef: hệ số ứng với loại xe\n
        '''
        self.id = id
        self.capacity = np.array(capacity)
        self.type = v_type
        self.coef = coef
    
    def __repr__(self):
        return "(" + str(self.id) + ")"

class Cluster:
    """
        Lớp chứa thông tin về 1 cụm, gồm: mảng sức chứa, mảng class City, số lượng items, mảng chứa trọng số hiện tại của cụm
    """
    def __init__(self, lat: float, lng:float, capacity, n_cities = 0, cities_id = [], current_mass = None, scale_coef = 0) -> None:

        '''
        Đặc tả về các trường thuộc tính: trong docs
        lat, lng: tọa độ của tâm cụm\n
        capacity: tải trọng của cụm (ứng với từng loại mặt hàng)\n

        '''
        self.lat = lat
        self.lng = lng
        self.capacity = np.array(capacity)
        self.n_cities = n_cities
        self.cities_id = list(cities_id)
        
        # if self.capacity == None: self.n_items = n_items
        if len(self.capacity.shape) == 1: self.n_items = self.capacity.shape[0]
        elif self.capacity == None: self.n_items = 0
        else: self.n_items = self.capacity.shape[1]

        self.scale_coef = scale_coef
        if current_mass is None:
            self.current_mass = np.zeros((self.n_items)) # np.array chứa khối lượng hiện tại của cluster đối với từng loại mặt hàng
        else: self.current_mass = np.array(current_mass)

    def get_center(self):
        return [self.lat, self.lng]

    def set_center(self, center):
        self.lat = center[0]
        self.lng = center[1]

    def append_city(self, city_id):
        self.cities_id.append(city_id)
        self.n_cities += 1
    
    def update_mass(self, add_mass, city_id):
        '''
        Khi có 1 node mới được đưa vào cluster thì ta cần update current mass của cluster, đồng thời gắn thêm id của node đó vào cities_id của cluster
        
        Params:\n
        `add_mass`: list hoặc array có len = n_items\n
        `city_id`: id của city được add\n
        '''
        self.current_mass+=np.array(add_mass)
        self.append_city(city_id)
    
    def clear_mass(self):
        self.current_mass = np.array(np.zeros(self.n_items))
        self.n_cities = 0
        self.cities_id= []

    def print(self, location_flag = False, capa_flag = False, current_mass_flag = False, get_n_cities_flag = False, city_id_list_flag = False, header = ''):
        if location_flag: print('{}Location: {}'.format(header, self.get_center()))
        if capa_flag: print('{}Capacity list: {}'.format(header,self.capacity))
        if current_mass_flag: print('{}Current mass: {}'.format(header,self.current_mass))
        if get_n_cities_flag: print('{}No. of customers: {}'.format(header,self.n_cities))
        if city_id_list_flag: print('{}Customers list: {}'.format(header, self.cities_id))