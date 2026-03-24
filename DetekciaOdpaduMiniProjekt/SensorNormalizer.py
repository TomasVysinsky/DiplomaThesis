from typing import List
import torch
import numpy as np
import hashlib
from functools import lru_cache
from RfidHelper import RFIDHelper

# ------- Konfiguračný blok (čo z objektu berieme a ako kódujeme) -------------
NUMERIC_FIELDS = [
    "axis_x_acc",  #Akcelerometer – zrýchlenie na osi X
    "axis_y_acc",  #Akcelerometer – zrýchlenie na osi Y
    "axis_z_acc",  #Akcelerometer – zrýchlenie na osi Z
    "sig_pwr",     #Radar – vzdialenosť v centimetroch
    #"b",           #Diagnostická hodnota – RFID teplota v °C
    "velocity",    #Rýchlosť vozidla
    "rssi_rfid"    #RFID – sila signálu snímania RFID tagu
]
INT_FIELDS = [
    #"a",           #Diagnostická hodnota – RFID status (error) kód
    #"c"            #Diagnostická hodnota – RFID teplota v °C
]                  # kódy (ponecháme ako čísla, normujeme)
BOOL_FIELDS = [
    #"gpio1",       #Výstupné GPIO pre svetelnú signalizáciu
    #"gpio2"        #Výstupné GPIO pre svetelnú signalizáciu
]  # do 0/1
# RFID: stačí binárna info či bol tag prítomný
EXTRA_BINARY = ["rfid_present"]   # odvodený stĺpec

ALL_FEATURES = NUMERIC_FIELDS + INT_FIELDS + BOOL_FIELDS + EXTRA_BINARY

class SensorNormalizer:
    def __init__(self, ignore_rfid : bool, use_left_right_arm_info : bool = True, field_names : List[str] = None):
        self.mean_ = None
        self.std_ = None
        self.ignore_rfid : bool = ignore_rfid
        self.use_left_right_arm_info : bool = use_left_right_arm_info
        self.field_names : List[str] = field_names
        print("SensorNormalizer: ignore_rfid =", ignore_rfid, ", use_left_right_arm_info =", use_left_right_arm_info, ", field_names =", field_names)
        
    @staticmethod
    def get_rfid_indexes() -> List[int]:
        #0 x
        #1 y
        #2 z
        #3 radar
        #4 rychlost
        #5 rfid sila -> rfid
        #6 hashovane rfid -> rfid
        #7 1 ak velky kontajner -> rfid
        #8 is dummy message
        return [5, 6, 7]

    def _extract_row(self, msg):
        # Ak sú explicitne zadané polia, použi len tie
        if self.field_names is not None:
            row = [float(getattr(msg, f)) for f in self.field_names]
            return np.array(row, dtype=np.float32)

        # binárna prítomnosť RFID tagu
        rfid_present = 0.0 if (msg.rfid_tag is None or msg.rfid_tag == "") else 1.0
        
        row = []
        for f in NUMERIC_FIELDS:
            if self.ignore_rfid and f == 'rssi_rfid' or f== 'b' : 
                row.append(0)
            else:
                if False: # f == 'sig_pwr':
                    row.append(0)
                else:
                    row.append(float(getattr(msg, f)))
        for f in INT_FIELDS:
            row.append(float(getattr(msg, f)))
        for f in BOOL_FIELDS:
            row.append(1.0 if getattr(msg, f) else 0.0)
            
        if self.ignore_rfid:
            debug = 1
            #row.append(0)
        else:
            rfid_float = SensorNormalizer.string_to_float_0_1(msg.rfid_tag) if not RFIDHelper.is_empty_rfid(msg.rfid_tag) else 0.0
            #print(rfid_float)
            row.append(rfid_float)
            #row.append(rfid_present)
            if self.use_left_right_arm_info:
                arm = 1.0 if (not RFIDHelper.is_empty_rfid(msg.rfid_tag) and RFIDHelper.is_big_container(msg.rfid_tag))  else 0.0
                row.append(arm)
        #print(len(row))
        return np.array(row, dtype=np.float32)

    def fit(self, messages):
        
        #Vypočíta mean a std z tréningovej množiny.
        
        matrix = np.stack([self._extract_row(m) for m in messages])
        self.mean_ = matrix.mean(axis=0)
        self.std_ = matrix.std(axis=0)
        # aby sme sa vyhli deleniu nulou:
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, messages) -> torch.Tensor:
        # Vráti torch.Tensor tvaru (N, D) so z‑skóre normovaním.
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Najskôr zavolaj fit() na tréningových dátach.")
        matrix = np.stack([self._extract_row(m) for m in messages])
        normed = (matrix - self.mean_) / self.std_
        return torch.from_numpy(normed)

    def transform_messages_with_dummy_flags(self, messages, dummy_flags, normalize: bool = True) -> torch.Tensor:
        """Batch transform messages and append dummy-flag column.

        Returns tensor of shape (N, D+1), same semantic output as repeated
        transform_message / numb_transform_message calls.
        """
        if len(messages) != len(dummy_flags):
            raise ValueError("messages and dummy_flags must have the same length")
        if len(messages) == 0:
            feature_count = len(NUMERIC_FIELDS) + len(INT_FIELDS) + len(BOOL_FIELDS)
            feature_count += 1 if self.ignore_rfid else (2 if self.use_left_right_arm_info else 1)
            return torch.empty((0, feature_count + 1), dtype=torch.float32)

        matrix = np.stack([self._extract_row(m) for m in messages])
        if normalize:
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("Najskôr zavolaj fit() na tréningových dátach.")
            matrix = (matrix - self.mean_) / self.std_

        dummy_col = np.asarray([1.0 if flag else 0.0 for flag in dummy_flags], dtype=np.float32).reshape(-1, 1)
        out = np.concatenate([matrix.astype(np.float32), dummy_col], axis=1)
        return torch.from_numpy(out)
    
    def transform_message(self, message, is_dummy : bool) -> torch.Tensor:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Najskôr zavolaj fit() na tréningových dátach.")
        message_tensor = self._extract_row(message)
        normed = (message_tensor - self.mean_) / self.std_
        normed = np.append(normed, 1.0 if is_dummy else 0.0)
        
        normed = np.array(normed, dtype=np.float32)
        return torch.from_numpy(normed)
    
    def numb_transform_message(self, message, is_dummy : bool) -> torch.Tensor:
        message_tensor = self._extract_row(message)
        message_tensor = np.append(message_tensor, 1.0 if is_dummy else 0.0)
        message_tensor = np.array(message_tensor, dtype=np.float32)
        return torch.from_numpy(message_tensor)
    
    @staticmethod
    @lru_cache(maxsize=200000)
    def string_to_float_0_1(s):
        # Vytvor hash stringu
        hash_bytes = hashlib.sha256(s.encode('utf-8')).digest()
        # Zober prvých 8 bajtov (64 bitov) a konvertuj na celé číslo
        int_value = int.from_bytes(hash_bytes[:8], byteorder='big')
        # Normalizuj na rozsah 0 až 1
        max_uint64 = 2**64 - 1
        return int_value / max_uint64
