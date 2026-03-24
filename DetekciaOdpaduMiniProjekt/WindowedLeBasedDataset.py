from bisect import bisect_left
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from DataModel import LitteringExecution, SensorDataMessage
from DataViewModel import LoadedDataViewModel, SensorDataMessageWithLittering, Vehicle
import RfidHelper

class SlidingWindow:
    def __init__(self):
        self.data_messages_left : List[SensorDataMessageWithLittering]
        self.data_messages_right : List[SensorDataMessageWithLittering]
        
        self.littering_execution : LitteringExecution
        
        self.tensor_input : torch.Tensor = None
        self.tensor_output : torch.Tensor = None
        self.result_class : int = None
        
    def start_time(self):
        if self.data_messages_left[0].data.real_time_computed < self.data_messages_right[0].data.real_time_computed:
            return self.data_messages_left[0].data.real_time_computed
        else:
            return self.data_messages_right[0].data.real_time_computed
    
    def end_time(self):
        last_index = len(self.data_messages_left) - 1
        if self.data_messages_left[last_index].data.real_time_computed > self.data_messages_right[last_index].data.real_time_computed:
            return self.data_messages_left[last_index].data.real_time_computed
        else:
            return self.data_messages_right[last_index].data.real_time_computed

class WindowedLeBasedDataset(Dataset):
    def __init__(
        self,
        vehicles : List[Vehicle],
        window_size : int,
        name : str,
        max_dlzka_trvania_le : float
    ):
        half_window_size : int = window_size  // 2
        
        self.samples = []
        self.windows_by_vehicle : Dict[str, List[SlidingWindow]] = {}
        
        print(f"Zostavenie datasetu pre {name} mnozinu")
        
        for vehicle in vehicles:
            print(vehicle.ecv)
            self.windows_by_vehicle[vehicle.ecv] = []
            left_unit = vehicle.unit_ids[0]
            right_unit = vehicle.unit_ids[1]
            
            les_left = vehicle.littering_executions_by_unit_id[left_unit]
            les_right = vehicle.littering_executions_by_unit_id[right_unit]
            
            #all_les = WindowedLeBasedDataset.merge_les(les_left, les_right)
            
            starts_left = starts = [msg.data.real_time_computed for msg in vehicle.data_message_with_literings_by_unit_id[left_unit]]
            starts_right = starts = [msg.data.real_time_computed for msg in vehicle.data_message_with_literings_by_unit_id[right_unit]]
            
            for unit_id in vehicle.unit_ids:
                for le in vehicle.littering_executions_by_unit_id[unit_id]:
                    if RfidHelper.RFIDHelper.is_empty_rfid(le.rfid_tag) and name == 'train':
                        continue
                    if (le.timestamp_end - le.timestamp_start).total_seconds() > max_dlzka_trvania_le:
                        continue
                    middle_time : datetime = le.timestamp_start + (le.timestamp_end - le.timestamp_start) / 2
                    start_time = middle_time - timedelta(seconds=(half_window_size/2))
                    
                    window = SlidingWindow()
                    window.data_messages_left = WindowedLeBasedDataset.extract_data(vehicle.data_message_with_literings_by_unit_id[left_unit], start_time, window_size, starts_left)
                    window.data_messages_right = WindowedLeBasedDataset.extract_data(vehicle.data_message_with_literings_by_unit_id[right_unit], start_time, window_size, starts_right)
                    window.littering_execution = le
                    self.windows_by_vehicle[vehicle.ecv].append(window)
                    
                    window_tensor_left = WindowedLeBasedDataset.get_window_as_tensor(window.data_messages_left, le)
                    window_tensor_right = WindowedLeBasedDataset.get_window_as_tensor(window.data_messages_right, le)
                    combined_windows = torch.cat([window_tensor_left, window_tensor_right], dim=0)
                    
                    y = torch.zeros(4)
                    window.result_class = WindowedLeBasedDataset.get_car_arm_as_int(le, les_left, les_right)
                    y[window.result_class] = 1.0
                    
                    window.tensor_input = combined_windows
                    #print(combined_windows.shape)
                    window.tensor_output = y
                    self.samples.append((combined_windows, y))
        print('Dataset vytvoreny')
                
    @staticmethod
    def _check_other_arm(le: LitteringExecution, other_execs: List[LitteringExecution], time_tolerance_ms: float = 1000) -> bool:
        """Vráti True ak sa v other_execs nachádza LE s časovo blízkym začiatkom aj koncom."""
        if not other_execs:
            return False
        tol = timedelta(milliseconds=time_tolerance_ms)
        starts = [e.timestamp_start for e in other_execs]
        left = bisect_left(starts, le.timestamp_start - tol)
        i = left
        while i < len(other_execs) and other_execs[i].timestamp_start <= le.timestamp_start + tol:
            e = other_execs[i]
            if abs(e.timestamp_start - le.timestamp_start) <= tol and abs(e.timestamp_end - le.timestamp_end) <= tol:
                return True
            i += 1
        return False

    @staticmethod
    def get_car_arm_as_int(
        le: LitteringExecution,
        les_left: List[LitteringExecution] = None,
        les_right: List[LitteringExecution] = None
    ) -> int:
        lower_arm = le.car_arm.upper()
        if lower_arm == 'L':
            return 3 if WindowedLeBasedDataset._check_other_arm(le, les_right) else 0
        elif lower_arm == 'R':
            return 3 if WindowedLeBasedDataset._check_other_arm(le, les_left) else 1
        elif lower_arm == 'B':
            return 2
        raise Exception(f'Problem car arm was {lower_arm}')
                
    @staticmethod
    def extract_data(data : List[SensorDataMessageWithLittering], start_time : datetime, window_size: int, starts):
        start_idx = None
        
        start_idx = bisect_left(starts, start_time)

        # ak nič nie je >= start_time, použijeme posledný záznam a nakopírujeme ho
        if start_idx is None:
            raise Exception("Problem")

        # vezmeme window_size záznamov od start_idx
        window = data[start_idx:start_idx + window_size]

        # ak je ich menej, doplníme posledný prvok
        if len(window) < window_size:
            last = window[-1]
            window.extend([last] * (window_size - len(window)))

        return window
            
            
                
    @staticmethod
    def merge_les(le_left: List[LitteringExecution], le_right : List[LitteringExecution]) -> List[LitteringExecution]:
        result : Dict[Tuple[datetime, datetime], LitteringExecution] = {}

        for le  in le_left + le_right:
            if le.trash_can == '' and RfidHelper.RFIDHelper.is_empty_rfid(le.rfid_tag):
                continue
            
            key = (le.timestamp_start, le.timestamp_end)
            if key not in result:
                result[key] = le
            else:
                le_existujuca = result[key]
                if le.trash_can != '':
                    if le_existujuca.trash_can == "SINGLE_BOTH":
                        le_existujuca.car_arm = 'B'
                    elif le_existujuca.trash_can == "DOUBLE":
                        le_existujuca.car_arm = 'B'
                    elif (le_existujuca.trash_can == "SINGLE_LEFT" and le.trash_can == "SINGLE_RIGHT") or (le_existujuca.trash_can == "SINGLE_RIGHT" and le.trash_can == "SINGLE_LEFT"):
                        le_existujuca.car_arm = 'B'
                        le_existujuca.trash_can = "SINGLE_BOTH"
                    else:
                        raise Exception(f"Trash can prva bol: {le.trash_can} trash can druha bol: {le_existujuca.trash_can}")
                else:
                    le_existujuca.car_arm = 'B'
                    if RfidHelper.RFIDHelper.is_big_container(le.rfid_tag):
                        le_existujuca.trash_can = "DOUBLE"
                    else:
                        le_existujuca.trash_can = "SINGLE_BOTH"
                
                
        return list(result.values())
    
    @staticmethod
    def get_window_as_tensor(okno : List[SensorDataMessageWithLittering], le : LitteringExecution) -> torch.Tensor:
        zoznam_tensorov = []
        for obj in okno:
            # pôvodná hodnota z input_tensor
            x = obj.input_tensor[0]
            y = obj.input_tensor[1]
            z = obj.input_tensor[2]
            radar = obj.input_tensor[3]
            #velocity = obj.input_tensor[5]
            

            # 1 ak timestamp patrí do intervalu, inak 0
            ts = obj.data.real_time_computed
            mask = 1.0 if le.timestamp_start <= ts <= le.timestamp_end else 0.0

            # vytvorenie 2D vektora: [povodna_hodnota, mask]
            zoznam_tensorov.append(
                torch.tensor([x, y, z, radar, mask], dtype=torch.float32)
            )

        window_tensor = torch.stack(zoznam_tensorov)  # shape: (window, 2)
        
        return torch.transpose(window_tensor, 0, 1)
                
     # -- PyTorch API ----------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]      # (tensor(40,12), tensor(1,))