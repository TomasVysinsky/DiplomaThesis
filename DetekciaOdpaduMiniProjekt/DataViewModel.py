from datetime import datetime, timedelta
import pickle
from DataModel import SensorDataMessage, LitteringExecution, WeightExecution
from typing import Dict, List
import torch
from joblib import dump, load
import math
import copy
import RfidHelper

from SensorNormalizer import SensorNormalizer
import bisect
from enum import Enum

class Vehicle():
    def __init__(self, ecv: str,
                 all_sensor_data_messages : List[SensorDataMessage],
                 all_littering_executions : List[LitteringExecution],
                 all_le_delta : List[LitteringExecution],
                 weight_executions : List[WeightExecution] = []):
        self.ecv : str = ecv
        self.sensor_data_messages : List[SensorDataMessage] = all_sensor_data_messages
        self.littering_executions : List[LitteringExecution] = all_littering_executions
        self.all_le_delta : List[LitteringExecution] = all_le_delta
        self.weight_executions : List[WeightExecution] = weight_executions
        
        self.data_message_with_literings_by_unit_id : Dict[str, List[SensorDataMessageWithLittering]] = {}
        self.littering_executions_by_unit_id : Dict[str, List[LitteringExecution]] = {}
        self.delta_le_by_unit_id : Dict[str, List[LitteringExecution]] = {}
        
        if 'video' in ecv:
            self.is_manually_anotated_vehicle : bool = True
        else:
            self.is_manually_anotated_vehicle : bool = False
        
        self.unit_ids = []
        
class SplittedVehicle(Vehicle):
    def __init__(self, 
                 ecv: str, 
                 all_sensor_data_messages: List[SensorDataMessage], 
                 all_littering_executions: List[LitteringExecution], 
                 all_le_delta : List[LitteringExecution],
                 split_time : datetime):
        super().__init__(ecv, all_sensor_data_messages, all_littering_executions, all_le_delta)
        self.split_time : datetime = split_time
        
class TestVehicleSplitComand():
    def __init__(self,
                 test_ration : float, 
                 ecv : str) -> None:
        self.test_ration : float = test_ration
        self.ecv : str = ecv
        
class ContainerType(Enum):
    NONE = 0
    SMALL = 1
    BIG = 2
        
class SensorDataMessageWithLittering():
    def __init__(self):
        self.data : SensorDataMessage = None
        self.littering_execution_id = -1
        self.input_tensor : torch.Tensor
        self.littering_execution_percentage : float = -1.
        self.container_type : ContainerType = ContainerType.NONE
        self.le_rfid = ''
        
        self.is_dummy_message = False
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'is_dummy_message' not in self.__dict__:
            self.is_dummy_message = False

class LoadedDataViewModel():
    def __init__(self, vehicles : List[Vehicle]) -> None:
        self.vehicles : List[Vehicle] = vehicles
        
        self.train_vehicles : List[Vehicle] = []
        self.test_vehicles : List[Vehicle] = []
        
        self.normalizer = SensorNormalizer(ignore_rfid=False)
        
        self.initialize()
    
    def get_video_vehicles(self, video_or_not_video : bool) -> List[Vehicle]:
        result = []
        for vehicle in self.vehicles:
            if video_or_not_video:
                if 'video' in vehicle.ecv:
                    result.append(vehicle)
            if not video_or_not_video:
                if 'video' not in vehicle.ecv:
                    result.append(vehicle)
                
        return result
    
    
        
    def save(self, file_name : str, use_joblib : bool = False):
        if use_joblib:
            dump(self.vehicles, file_name + '_vehicles.joblib')
            dump(self.train_vehicles, file_name +  '_train_vehicles.joblib')
            dump(self.test_vehicles, file_name + '_test_vehicles.joblib')
            
        else:
            with open(file_name + "_vehicles.pcl", "wb") as f:
                pickle.dump(self.vehicles, f)
                
            with open(file_name + "_train_vehicles.pcl", "wb") as f:
                pickle.dump(self.train_vehicles, f)
            
            with open(file_name + "_test_vehicles.pcl", "wb") as f:
                pickle.dump(self.test_vehicles, f)
            
    def load(self, file_name : str, 
             use_job_lib : bool = False, 
             load_vehicles : bool = True, 
             load_train_and_test : bool = False):
        print("Spustam nacitanie dat")
        if use_job_lib:
            if load_vehicles:
                self.vehicles = load(file_name + '_vehicles.joblib')
            if load_train_and_test:
                self.train_vehicles = load(file_name + '_train_vehicles.joblib')
                self.test_vehicles = load(file_name + '_test_vehicles.joblib')
        else:
            if load_vehicles:
                print("Spúšťam načítanie loaded data viewmodelu")
                with open(file_name + "_vehicles.pcl", "rb") as f:
                    self.vehicles = pickle.load(f)
            
            if load_train_and_test:
                with open(file_name + "_train_vehicles.pcl", "rb") as f:
                    self.train_vehicles = pickle.load(f)
                
                with open(file_name + "_test_vehicles.pcl", "rb") as f:
                    self.test_vehicles = pickle.load(f)
        print("Nacitanie dokoncene")
        
    @staticmethod
    def add_ids_to_littering_executions(vehicles : List[Vehicle]):
        id = 0
        for vehicle in vehicles:
            for unit_id in vehicle.unit_ids:
                for le in vehicle.littering_executions_by_unit_id[unit_id]:
                    if le.is_original_prediction:
                        le.id = id
                        id += 1
        
    @staticmethod
    def add_dummy_messages(vehicles : List[Vehicle]):
        print("Pridavam dummy messages")
        for vehicle in vehicles:
            for unit_id in vehicle.unit_ids:
                data_with_dummy_messages = []
                
                original_data = vehicle.data_message_with_literings_by_unit_id[unit_id]
                
                for i in range(len(original_data) - 1):
                    first = original_data[i]
                    second = original_data[i+1]
                    
                    data_with_dummy_messages.append(first)
                    
                    
                    total_seconds = (second.data.real_time_computed - first.data.real_time_computed).total_seconds()
                    if total_seconds < 10.0 and total_seconds > 0.75:
                        #print(f'total seconds: {total_seconds}')
                        total_half_seconds_interval = math.floor(total_seconds / 0.5)
                        for j in range(total_half_seconds_interval): 
                            new_time = first.data.real_time_computed + timedelta(seconds= (0.5 * (j+1)))
                            
                            #print(f"pre bus : {vehicle.ecv} unit_id: {unit_id} bola pridana message s casom : {new_time} first: {first.data.real_time_computed} second: {second.data.real_time_computed}")
                            
                            deep_copy_message = copy.deepcopy(first)
                            deep_copy_message.is_dummy_message = True
                            deep_copy_message.data.real_time_computed = new_time
                            data_with_dummy_messages.append(deep_copy_message)
                        
                data_with_dummy_messages.append(original_data[-1])
                
                vehicle.data_message_with_literings_by_unit_id[unit_id] = data_with_dummy_messages
        print("Hotovo")
        
    @staticmethod
    def split_b_les_between_car_arms(vehicles: List[Vehicle]):
        for vehicle in vehicles:
            splitted = 0
            
            for i in range(len(vehicle.unit_ids)):
                unit_id = vehicle.unit_ids[i]
                other_unit_id = vehicle.unit_ids[0] if i == 1 else vehicle.unit_ids[1]
                
                current_les = vehicle.littering_executions_by_unit_id[unit_id]
                for le in current_les:
                    if le.is_original_prediction and ( le.car_arm == 'b' or le.car_arm == 'B'):
                        #le.car_arm = 'L'
                        le_new = LitteringExecution()
                        le_new.trash_can = le.trash_can
                        le_new.id = le.id
                        le_new.is_delta_rfid = le.is_delta_rfid
                        le_new.unit_id = other_unit_id
                        le_new.timestamp_start = le.timestamp_start
                        le_new.timestamp_end = le.timestamp_end
                        le_new.car_arm = 'B'
                        le_new.is_original_prediction = False
                        
                        splitted +=1
                        
                        le_new.rfid_tag = le.rfid_tag
                        
                        le_new.is_paired_to_prediction = le.is_paired_to_prediction
                        
                        vehicle.littering_executions_by_unit_id[other_unit_id].append(le_new)
                    elif not (le.car_arm in ['B', 'b', 'L', 'l', 'R', 'r']):
                        raise Exception(f"car arm was: {le.car_arm}")
        
            for unit_id in vehicle.unit_ids:
                sorted_littering_e = sorted(vehicle.littering_executions_by_unit_id[unit_id], key=lambda obj : obj.timestamp_start)
                vehicle.littering_executions_by_unit_id[unit_id] = sorted_littering_e
                
            for unit_id in vehicle.unit_ids:
                to_remove = []
                
                les_for_unit_id = vehicle.littering_executions_by_unit_id[unit_id]
                for i in range(len(les_for_unit_id)-1):
                    le_first = les_for_unit_id[i]
                    le_second = les_for_unit_id[i+1]
                    
                    if le_first.timestamp_start == le_second.timestamp_start and le_first.timestamp_end == le_second.timestamp_end:
                        print(f'vehicle: {vehicle.ecv} unit id: {le_first.unit_id} was removed, second unit id: {le_second.unit_id} id : {le_second.id} ')
                        print(f"start: {le_first.timestamp_start} end : {le_second.timestamp_end} first car arm: {le_first.car_arm} second car arm: {le_second.car_arm}")
                        if not le_first.is_original_prediction:
                            splitted-=1
                        to_remove.append(le_first)
                
                les_for_unit_id = [le for le in les_for_unit_id if le not in to_remove]
                vehicle.littering_executions_by_unit_id[unit_id] = les_for_unit_id
            #print(f"Ecv: {vehicle.ecv} Splitted count: {splitted}")
            
        for vehicle in vehicles:
            for unit_id in vehicle.unit_ids:
                sorted_littering_e = sorted(vehicle.littering_executions_by_unit_id[unit_id], key=lambda obj : obj.timestamp_start)
                vehicle.littering_executions_by_unit_id[unit_id] = sorted_littering_e
                
    @staticmethod
    def remove_les_from_vehicle(vehicles : List[Vehicle], ecv : str, to_remove : List[int]):
        vehicle = LoadedDataViewModel.get_vehicle_by_ecv(vehicles, ecv)
        if vehicle is not None:
            for unit_id in vehicle.unit_ids:
                data = vehicle.littering_executions_by_unit_id[unit_id]
                for i in range(len(data)-1, -1, -1):
                    
                    le = data[i]
                    if not le.is_delta_rfid  and \
                        (le.id in to_remove):
                            print(f'le for vehicle {vehicle.ecv} id: {le.id} bola odstranena v ramci manualneho preprocesingu (prekryvala sa s inou)')
                            del data[i]
                            
    @staticmethod
    def remove_delta_les_from_vehicle(vehicles : List[Vehicle], ecv : str, to_remove : List[int]):
        vehicle = LoadedDataViewModel.get_vehicle_by_ecv(vehicles, ecv)
        if vehicle is not None:
            for unit_id in vehicle.unit_ids:
                data = vehicle.littering_executions_by_unit_id[unit_id]
                for i in range(len(data)-1, -1, -1):
                    
                    le = data[i]
                    if le.is_delta_rfid  and \
                        (le.id in to_remove):
                            #print(le.id)
                            del data[i]
    
    

    @staticmethod
    def remove_bad_littering_e_and_merge(vehicles : List[Vehicle], seconds_to_merge = 20):
        
        LoadedDataViewModel.remove_les_from_vehicle(vehicles, 'ZA127IR', [1164])
        LoadedDataViewModel.remove_les_from_vehicle(vehicles, 'ZA499JN', [68, 711, 706, 799, 1478, 1651, 1383, 679, 1418])
        LoadedDataViewModel.remove_les_from_vehicle(vehicles, 'ZA346KA', [289])
        LoadedDataViewModel.remove_les_from_vehicle(vehicles, 'ZA234JG', [1047, 1052])
        
        
        to_remove = [22, 20, 19, 7, 13, 9, 23, 12, 15, 10, 4, 6, 34, 50, 55, 51, 52, 45, 44, 42, 40, 41, 56, 38, 32, 26, 33, 59, 37, 39, 30, 28, 27]
        LoadedDataViewModel.remove_delta_les_from_vehicle(vehicles, 'ZA127IR', to_remove)
        
        to_remove = [4, 9, 30, 14, 28, 19, 7, 13, 22, 25, 12, 18, 8, 5, 3, 1, 33]
        LoadedDataViewModel.remove_delta_les_from_vehicle(vehicles, 'ZA234JG', to_remove)
        
        to_remove = [3, 4, 5, 6, 7]
        LoadedDataViewModel.remove_delta_les_from_vehicle(vehicles, 'ZA255KC', to_remove)
        
        to_remove = [1, 2, 6, 7]
        LoadedDataViewModel.remove_delta_les_from_vehicle(vehicles, 'ZA346KA', to_remove)
         
        to_remove = [1, 2, 4, 6, 5]
        LoadedDataViewModel.remove_delta_les_from_vehicle(vehicles, 'ZA196JN_video', to_remove)               
         
        to_remove = [21, 12, 16, 3, 18, 2, 7, 4, 17, 9, 23, 14, 6, 8, 1]
        LoadedDataViewModel.remove_delta_les_from_vehicle(vehicles, 'ZA234JG_video', to_remove) 
        
        to_remove = [2, 4, 1, 3]
        LoadedDataViewModel.remove_delta_les_from_vehicle(vehicles, 'ZA503JU_video', to_remove)
        
        
        for vehicle in vehicles:
            for unit_id in vehicle.unit_ids:
                les_for_unit_id = vehicle.littering_executions_by_unit_id[unit_id]
                
                to_remove = []
                
                #remove 0 seconds long
                for le in les_for_unit_id:
                    if le.timestamp_start == le.timestamp_end:
                        to_remove.append(le)
                        print(f"vehicle {vehicle.ecv} unit id: {le.unit_id} id: {le.id} Removed le because 0 length")
                if len(to_remove) > 0:
                    les_for_unit_id = [le for le in les_for_unit_id if le not in to_remove]
                    vehicle.littering_executions_by_unit_id[unit_id] = les_for_unit_id
                    
                    
                #remove 
                to_remove = set()
                literings_on_unit_id = vehicle.littering_executions_by_unit_id[unit_id]
                
                for i in range(len(literings_on_unit_id) -1) :
                    previous = literings_on_unit_id[i]
                    next = literings_on_unit_id[i+1]
                    
                    duration_seconds = (next.timestamp_start - previous.timestamp_end).total_seconds()
                    if duration_seconds <= 0.0:
                        
                        maly_velky = ''
                        if not RfidHelper.RFIDHelper.is_empty_rfid(previous.rfid_tag) and not RfidHelper.RFIDHelper.is_empty_rfid(next.rfid_tag):
                            prvy_velky = RfidHelper.RFIDHelper.is_big_container(previous.rfid_tag)
                            druhy_velky = RfidHelper.RFIDHelper.is_big_container(next.rfid_tag)
                            if (prvy_velky and not druhy_velky) or (not prvy_velky and druhy_velky):
                                maly_velky = 'ano'
                            else:
                                maly_velky = 'nie'
                        else:
                            if not RfidHelper.RFIDHelper.is_empty_rfid(previous.rfid_tag) and RfidHelper.RFIDHelper.is_big_container(previous.rfid_tag):
                                maly_velky = 'prvy velky, druhy bez'
                                print(f"Vysyp {vehicle.ecv} s id: {next.id} bol odstraneny, lebo sa prekryva s inym vysypom, ktory ma unit id")
                                to_remove.add(next.id)
                                
                            elif not RfidHelper.RFIDHelper.is_empty_rfid(next.rfid_tag) and RfidHelper.RFIDHelper.is_big_container(next.rfid_tag):
                                maly_velky = 'druhy velky, prvy bez'
                                print(f"Vysyp {vehicle.ecv} s id: {previous.id} bol odstraneny, lebo sa prekryva s inym vysypom, ktory ma unit id")
                                to_remove.add(previous.id)
                                
                            else:
                                #
                                print("problem, oba nemaju rfid")
                                #raise Exception("problem")
                                maly_velky = 'oba nemaju rfid'
                les_for_unit_id = [x for x in vehicle.littering_executions_by_unit_id[unit_id] if x.id not in to_remove] 
                
                
                #spojenie doklepov
                removed = False
                last_non_removed : LitteringExecution = None
                to_remove = []
                for i in range(len(les_for_unit_id)-1):
                    le_first = les_for_unit_id[i]
                    le_second = les_for_unit_id[i+1]
                    
                    if not RfidHelper.RFIDHelper.is_empty_rfid(le_first.rfid_tag) and le_first.rfid_tag == le_second.rfid_tag \
                        and (le_second.timestamp_start - le_first.timestamp_end).total_seconds() < seconds_to_merge:
                            
                            if not removed:
                                last_non_removed = le_first
                            last_non_removed.timestamp_end = le_second.timestamp_end
                            last_non_removed.additional_info = f' doklep do {le_second.id}'
                            #le_second.additional_info = f'vymazana kvoli {last_non_removed.id}'
                            to_remove.append(le_second)
                            print(f"vehicle {vehicle.ecv} -repeated lifting, removed because less than {seconds_to_merge} seconds and rfid : {le_first.rfid_tag}, id: {le_second.id} unit id : {le_second.unit_id}")
                            removed = True
                    else:
                        removed = False
                        last_non_removed = None
                        
                if len(to_remove) > 0:
                    debug = 1
                les_for_unit_id = [le for le in les_for_unit_id if le not in to_remove]
                vehicle.littering_executions_by_unit_id[unit_id] = les_for_unit_id
                
            
    
    @staticmethod
    def get_vehicle_by_ecv(vehicles: List[Vehicle], ecv: str) -> Vehicle:
        vehicle = next((v for v in vehicles if v.ecv == ecv), None)
        return vehicle
        
    def initialize(self):
        
        sensor_data_messages_count = 0
        littering_executions_count = 0
        for vehicle in self.vehicles:
            sensor_data_messages_count += len(vehicle.sensor_data_messages)
            littering_executions_count += len(vehicle.littering_executions)
        
        print("Bola spustena inicializacia dat")
        print(f"Celkovy pocet data messages : {sensor_data_messages_count}")
        print(f"Celkovy pocet vysypov: {littering_executions_count}")
        
        for vehicle in self.vehicles:
            
            for message in vehicle.sensor_data_messages:
                if message.unit_id not in vehicle.data_message_with_literings_by_unit_id:
                    vehicle.data_message_with_literings_by_unit_id[message.unit_id] = []
                
                message_with_littering = SensorDataMessageWithLittering()
                message_with_littering.data = message
                
                vehicle.data_message_with_literings_by_unit_id[message.unit_id].append(message_with_littering)
                
            for literring_e in vehicle.littering_executions:
                if literring_e.unit_id not in vehicle.littering_executions_by_unit_id:
                    vehicle.littering_executions_by_unit_id[literring_e.unit_id] = []
                if literring_e.unit_id not in vehicle.delta_le_by_unit_id:
                    vehicle.delta_le_by_unit_id[literring_e.unit_id] = []
                vehicle.littering_executions_by_unit_id[literring_e.unit_id].append(literring_e)
                
                
                
            for litering_e in vehicle.all_le_delta:
                vehicle.delta_le_by_unit_id[litering_e.unit_id].append(litering_e)
                
            print(f"Data pre vozidlo {vehicle.ecv} obsahuju nasledovne unit id:")
            for unit_id in vehicle.data_message_with_literings_by_unit_id:
                vehicle.unit_ids.append(unit_id)
                
                count_data_messages = len(vehicle.data_message_with_literings_by_unit_id[unit_id])
                
                if unit_id not in vehicle.littering_executions_by_unit_id:
                    vehicle.littering_executions_by_unit_id[unit_id] = []
                count_littering_e = len(vehicle.littering_executions_by_unit_id[unit_id])
                print(f"unit id: {unit_id} pocet sensor messages : {count_data_messages} pocet vysypov: {count_littering_e}")
            
            for unit_id in vehicle.littering_executions_by_unit_id:
                if unit_id not in vehicle.unit_ids:
                    raise ValueError(f"Littering execution pre ecv: {vehicle.ecv} unit id: {unit_id}, ktore nie je v datach")
                
            #if not self.check_littering_e_uniq_ids(vehicle):
                #raise ValueError("Niektore littering e nemaju unikatne idcka")
                
            for unit_id in vehicle.unit_ids:
                sorted_messages = sorted(vehicle.data_message_with_literings_by_unit_id[unit_id], key=lambda obj: obj.data.real_start_time())
                vehicle.data_message_with_literings_by_unit_id[unit_id] = sorted_messages
                
                sorted_littering_e = sorted(vehicle.littering_executions_by_unit_id[unit_id], key=lambda obj : obj.timestamp_start)
                vehicle.littering_executions_by_unit_id[unit_id] = sorted_littering_e
                
                if unit_id in vehicle.delta_le_by_unit_id:
                    sorted_delta_les = sorted(vehicle.delta_le_by_unit_id[unit_id], key=lambda obj : obj.timestamp_start)
                    vehicle.delta_le_by_unit_id[unit_id] = sorted_delta_les
                else:
                    vehicle.delta_le_by_unit_id[unit_id] = []
                    
        print("split b les between car arms")
        LoadedDataViewModel.split_b_les_between_car_arms(self.vehicles) 
        print('remove bad les')
        LoadedDataViewModel.remove_bad_littering_e_and_merge(self.vehicles)  
          
        for vehicle in self.vehicles:
            for unit_id in vehicle.unit_ids:
                
                littering_executions = vehicle.littering_executions_by_unit_id[unit_id]
                data_messages_with_littering = vehicle.data_message_with_literings_by_unit_id[unit_id]
                times = [m.data.real_time_computed for m in data_messages_with_littering]
                
                for littering_execution in littering_executions:
                    # data_messages_with_littering are sorted by data.real_time_computed
                    start_idx = bisect.bisect_left(times, littering_execution.timestamp_start)
                    end_idx = bisect.bisect_right(times, littering_execution.timestamp_end)

                    for msg in data_messages_with_littering[start_idx:end_idx]:
                        msg.littering_execution_id = littering_execution.id
                        
                        
                        if littering_execution.car_arm in ['B', 'b']:
                            msg.container_type = ContainerType.BIG 
                        elif littering_execution.car_arm in ['L', 'l', 'R', 'r']:
                            msg.container_type = ContainerType.SMALL
                        else:
                            raise Exception(f"Unsupported car arm type: {littering_execution.car_arm}")
                        msg.le_rfid = littering_execution.rfid_tag
                        
                        start_t = littering_execution.timestamp_start
                        end_t = littering_execution.timestamp_end
                        mid_t = start_t + (end_t - start_t) / 2

                        # distance from the middle, normalized to [0..1] where 1 == at start/end, 0 == at middle
                        total_sec = max((end_t - start_t).total_seconds(), 1e-9)
                        dist_mid_norm = abs((msg.data.real_time_computed - mid_t).total_seconds()) / (total_sec / 2)

                        # score: 1 at middle, linearly decreases to 0 at edges (and stays 0 outside)
                        msg.littering_execution_percentage = max(0.0, 1.0 - dist_mid_norm)
                        
            
            # identifikacia medzier
            if False:
                print(f"Medzery v message data pre vozidlo {vehicle.ecv}:")
                for unit_id in vehicle.unit_ids:
                    messages = vehicle.data_message_with_literings_by_unit_id[unit_id]
                    last_time : datetime = None
                    for message in messages:
                        curr_time = message.data.real_time_computed
                        if last_time is not None and (curr_time - last_time).total_seconds() > 2:
                            medzera = (curr_time - last_time).total_seconds()
                            print(f"unit id: {unit_id} cas od: {last_time} cas do : {curr_time} sekund: {medzera}")
                        last_time = curr_time
            print("------------------------")
        debug = 1
        
    @staticmethod
    def get_split_point(vehicle: Vehicle, test_ratio : float) -> datetime:
        min_length = datetime.max
        min_unit_id : str = ''
        
        for unit_id in vehicle.unit_ids:
            last_data = vehicle.data_message_with_literings_by_unit_id[unit_id][-1]
            if last_data.data.real_time_computed < min_length:
                min_length = last_data.data.real_time_computed
                min_unit_id = unit_id
        
        data_list = vehicle.data_message_with_literings_by_unit_id[min_unit_id]  
                
        split_point = int(len(data_list) * (1 - test_ratio))
        split_point = min(split_point, len(data_list) - 1) # fix for the case when test_ratio == 0.0
        split_point_time = data_list[split_point].data.real_time_computed
        return split_point_time

    @staticmethod
    def get_all_vehicles_data(vehicles : List[Vehicle]) -> List[SensorDataMessageWithLittering]:
        all_messages : List[SensorDataMessageWithLittering] = []
        for train_vehicle in vehicles:
            for unit_id in train_vehicle.unit_ids:
                data_all = train_vehicle.data_message_with_literings_by_unit_id[unit_id]
                for data in data_all:
                    all_messages.append(data)
                    
        return all_messages
    
    
    @staticmethod
    def remove_time_interval_from_vehicle(vehicle: Vehicle, start_time: datetime, end_time : datetime):
        for unit_id in vehicle.unit_ids:
            #remove SensorDataMessageWithLittering
            original_list = vehicle.data_message_with_literings_by_unit_id[unit_id]
            filtered_list = [
                w for w in original_list
                if not (start_time <= w.data.real_time_computed <= end_time)
            ]
            vehicle.data_message_with_literings_by_unit_id[unit_id] = filtered_list

            # remov LitteringExecution (main)
            original_le = vehicle.littering_executions_by_unit_id[unit_id]
            filtered_le = [
                le for le in original_le
                if not (start_time <= le.timestamp_start <= end_time)
            ]
            vehicle.littering_executions_by_unit_id[unit_id] = filtered_le

            # remove LitteringExecution (delta)
            original_delta = vehicle.delta_le_by_unit_id[unit_id]
            filtered_delta = [
                le for le in original_delta
                if not (start_time <= le.timestamp_start <= end_time)
            ]
            vehicle.delta_le_by_unit_id[unit_id] = filtered_delta
        
        
        
    def split_to_train_and_test(self, vehicle_split_command : List[TestVehicleSplitComand], normalize_values=True):
        self.train_vehicles  = []
        self.test_vehicles = []

        command_by_ecv: Dict[str, TestVehicleSplitComand] = {
            cmd.ecv: cmd for cmd in vehicle_split_command
        }

        train_messages_all: List[SensorDataMessageWithLittering] = []
        test_messages_all: List[SensorDataMessageWithLittering] = []
        
        for vehicle in self.vehicles:
            vehicle_command = command_by_ecv.get(vehicle.ecv)
            if vehicle_command is None:
                self.train_vehicles.append(vehicle)
                for unit_id in vehicle.unit_ids:
                    train_messages_all.extend(vehicle.data_message_with_literings_by_unit_id[unit_id])
                print(f'Vozidlo s ecv: {vehicle.ecv} bolo zaradene do trenovacej mnoziny')
            elif vehicle_command is not None and vehicle_command.test_ration == 1.0:
                self.test_vehicles.append(vehicle)
                for unit_id in vehicle.unit_ids:
                    test_messages_all.extend(vehicle.data_message_with_literings_by_unit_id[unit_id])
                print(f'Vozidlo s ecv: {vehicle.ecv} bolo zaradene do testovacej mnoziny')
            else:
                
                split_point_time = LoadedDataViewModel.get_split_point(vehicle, vehicle_command.test_ration)
                
                test_vehicle  = SplittedVehicle(vehicle.ecv, [], [], [], split_point_time)
                train_vehicle = SplittedVehicle(vehicle.ecv, [], [], [], split_point_time)
                
                self.train_vehicles.append(train_vehicle)
                self.test_vehicles.append(test_vehicle)
                
                for unit_id in vehicle.unit_ids:
                    test_vehicle.unit_ids.append(unit_id)
                    train_vehicle.unit_ids.append(unit_id)
                    
                    train_vehicle.data_message_with_literings_by_unit_id[unit_id] = []
                    train_vehicle.littering_executions_by_unit_id[unit_id] = []
                    
                    test_vehicle.data_message_with_literings_by_unit_id[unit_id] = []
                    test_vehicle.littering_executions_by_unit_id[unit_id] = []

                    unit_messages = vehicle.data_message_with_literings_by_unit_id[unit_id]
                    unit_timestamps = [d.data.real_time_computed for d in unit_messages]
                    split_idx = bisect.bisect_left(unit_timestamps, split_point_time)

                    train_messages = unit_messages[:split_idx]
                    test_messages = unit_messages[split_idx:]

                    train_vehicle.data_message_with_literings_by_unit_id[unit_id] = train_messages
                    test_vehicle.data_message_with_literings_by_unit_id[unit_id] = test_messages

                    train_messages_all.extend(train_messages)
                    test_messages_all.extend(test_messages)

                    unit_les = vehicle.littering_executions_by_unit_id[unit_id]
                    train_vehicle.littering_executions_by_unit_id[unit_id] = [
                        le for le in unit_les if le.timestamp_start < split_point_time
                    ]
                    test_vehicle.littering_executions_by_unit_id[unit_id] = [
                        le for le in unit_les if le.timestamp_end > split_point_time
                    ]
                            
                print(f"Data boli rozdelene na trenovaciu a testovaciu mnozinu pre vozidlo: {vehicle.ecv}")
                print(f"Cas rozdelenia: {split_point_time}")
                for unit_id in vehicle.unit_ids:
                    print(f"Unit id: {unit_id} pocet train: {len(train_vehicle.data_message_with_literings_by_unit_id[unit_id])} pocet test: {len(test_vehicle.data_message_with_literings_by_unit_id[unit_id])}")
                  
       
                    
        self.normalizer.fit([obj.data for obj in train_messages_all])

        train_tensors = self.normalizer.transform_messages_with_dummy_flags(
            [obj.data for obj in train_messages_all],
            [obj.is_dummy_message for obj in train_messages_all],
            normalize=normalize_values,
        )
        for i, data in enumerate(train_messages_all):
            data.input_tensor = train_tensors[i].clone()

        test_tensors = self.normalizer.transform_messages_with_dummy_flags(
            [obj.data for obj in test_messages_all],
            [obj.is_dummy_message for obj in test_messages_all],
            normalize=normalize_values,
        )
        for i, data in enumerate(test_messages_all):
            data.input_tensor = test_tensors[i].clone()

        
        print('Normalizer parameters: ')
        print(self.normalizer.mean_)
        print(self.normalizer.std_)
            
        print("------------------------------------")
        debug = 1
        
    def add_tensor_to_all_data(self, normalize_values : bool = True):
        self.normalizer = SensorNormalizer(False)
        data_messages_all = LoadedDataViewModel.get_all_vehicles_data(self.vehicles)
        self.normalizer.ignore_rfid = False
        self.normalizer.fit([obj.data for obj in data_messages_all])
        
        norm_fc = self.normalizer.transform_message if normalize_values else self.normalizer.numb_transform_message

        for data in data_messages_all:
            data.input_tensor = norm_fc(data.data, data.is_dummy_message)

        
        
    def check_littering_e_uniq_ids(self, vehicle : Vehicle) -> bool:
        all_ids = [obj.id for obj in vehicle.littering_executions]
        unique_ids_set = set(all_ids)
        return len(all_ids) == len(unique_ids_set)

