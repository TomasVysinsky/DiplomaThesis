import csv
import os
from DataModel import LitteringExecution, WeightExecution, SensorDataMessage, SensorHeaderMessage, SmallLitteringExecution, SmallSensorDataMessage, VideoLitering
from datetime import datetime, timedelta
from typing import Dict, List
from DataViewModel import LoadedDataViewModel, Vehicle
import DataViewModel
import vehicle_helper
import RfidHelper

class DataLoaderProprietary:
    def __init__(self):
        self.data_folder_path : str = 'data/'
        self.small_start_dataset_path : str = 'D1.1_Počiatočný malý dataset/'
        self.dataset_june_path : str = 'dataset_200625/'
        
    @staticmethod 
    def parse_datetime(value) -> datetime:
        result = None
        value = value.replace('"', '')
        try:
            result = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except:
            try:
                result = datetime.strptime(value, '%Y-%m-%d %H:%M:%S.%f')
            except:
                result = datetime.strptime(value, '%d.%m.%Y %H:%M')
        return result
    
    @staticmethod
    def load_all_data(load_from_csv_files : bool, file_name : str, load_video_annotations : bool, video_only_vehicles : bool = False):

        if load_from_csv_files:
          pass
        else:
            loaded_data_view_model : DataViewModel.LoadedDataViewModel = DataViewModel.LoadedDataViewModel([])
            loaded_data_view_model.load(file_name, False, True, False)
            return DataLoaderProprietary.adjust_loaded_data_viewmodel(loaded_data_view_model)
        
    @staticmethod
    def remove_delta_les_from_vehicle(loaded_data_view_model: LoadedDataViewModel, ecv : str, to_remove : List[int]):
        vehicle = loaded_data_view_model.get_vehicle_by_ecv(loaded_data_view_model.vehicles, ecv)
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
    def remove_les_from_vehicle(loaded_data_view_model: LoadedDataViewModel, ecv : str, to_remove : List[int]):
        vehicle = loaded_data_view_model.get_vehicle_by_ecv(loaded_data_view_model.vehicles, ecv)
        if vehicle is not None:
            for unit_id in vehicle.unit_ids:
                data = vehicle.littering_executions_by_unit_id[unit_id]
                for i in range(len(data)-1, -1, -1):
                    
                    le = data[i]
                    if not le.is_delta_rfid  and \
                        (le.id in to_remove):
                            print(le.id)
                            del data[i]
        
    @staticmethod 
    def adjust_loaded_data_viewmodel(loaded_data_view_model : LoadedDataViewModel) -> LoadedDataViewModel:
        for vehicle in loaded_data_view_model.vehicles:
            if vehicle_helper.VehicleHelper.get_car_arm(vehicle.ecv, vehicle.unit_ids[0]) == 'R':
                unit_id = vehicle.unit_ids[0]
                vehicle.unit_ids[0] = vehicle.unit_ids[1]
                vehicle.unit_ids[1] = unit_id
        
        #prehodenie
        id_prve = -1
        id_druhe = -1

        i = 0
        for vehicle in loaded_data_view_model.vehicles:
            if vehicle.ecv == 'ZA234JG_video':
                id_prve = i
            elif vehicle.ecv == 'ZA196JN_video':
                id_druhe = i
            i+=1
            
        prve = loaded_data_view_model.vehicles[id_prve]
        loaded_data_view_model.vehicles[id_prve] = loaded_data_view_model.vehicles[id_druhe]
        loaded_data_view_model.vehicles[id_druhe] = prve
        
        ecv = 'ZA127IR'
        vehicle = loaded_data_view_model.get_vehicle_by_ecv(loaded_data_view_model.vehicles, ecv)

        for unit_id in vehicle.unit_ids:
            is_first_unit_id = unit_id == vehicle.unit_ids[0]
            if is_first_unit_id:
                for le in vehicle.littering_executions_by_unit_id[unit_id]:
                    ts= le.timestamp_start
                    if ts.date() == datetime(2025, 5, 28).date():
                        le.timestamp_start =  le.timestamp_start + timedelta(seconds=4) 
                        le.timestamp_end = le.timestamp_end + timedelta(seconds=4)
                        
        

        for unit_id in vehicle.unit_ids:
            is_first_unit_id = unit_id == vehicle.unit_ids[0]
            if is_first_unit_id:
                for data in vehicle.data_message_with_literings_by_unit_id[unit_id]:
                    ts= data.data.real_time_computed
                    if ts.date() == datetime(2025, 5, 28).date():
                        data.data.real_time_computed = ts + timedelta(seconds=4) 
        
        
        LoadedDataViewModel.add_ids_to_littering_executions(loaded_data_view_model.vehicles)
        return loaded_data_view_model
