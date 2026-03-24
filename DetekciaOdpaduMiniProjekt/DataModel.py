from typing import Optional, List
from datetime import datetime, timedelta

class SensorHeaderMessage:
    def __init__(self):
        self.id: int  # db id
        self.unit_id: str  # Identifikátor IOT zariadenia
        self.msg_id: int  # Poradové číslo správy
        self.msg_type: str  # Typ správy – sensor_status
        self.timestamp: datetime  # Časová pečiatka v čase odoslania dát – formát: “yyyy-MM-dd HH:mm:ss.SSS”
        self.lat: float  # Zemepisná šírka
        self.lng: float  # Zemepisná dĺžka
        self.processed: bool  # Príznak spracovania údajov
        self.processed_new: Optional[bool]  # Nepoužíva sa
        self.device_timestamp: datetime  # Časová pečiatka IOT zariadenia
        self.server_timestamp: datetime  # Časová pečiatka servera
        
class SmallSensorDataMessage():
    def __init__(self):
        self.id: int  # db id
        self.unit_id: str  # Identifikátor IOT zariadenia
        self.axis_x_acc: float  # Akcelerometer – zrýchlenie na osi X
        self.axis_y_acc: float  # Akcelerometer – zrýchlenie na osi Y
        self.axis_z_acc: float  # Akcelerometer – zrýchlenie na osi Z
        self.sig_pwr: float  # Radar – vzdialenosť v centimetroch
        self.a: int  # Diagnostická hodnota – RFID status (error) kód
        self.b: float  # Diagnostická hodnota – RFID teplota v °C
        self.c: int  # Diagnostická hodnota – Radar reinit
        self.gpio1: bool  # Výstupné GPIO pre svetelnú signalizáciu
        self.gpio2: bool  # Výstupné GPIO pre svetelnú signalizáciu
        self.rel_time: int  # Relatívna hodnota voči časovej pečiatke v milisekundách – timestamp + rel_time = skutočný čas merania
        self.timestamp: datetime  # Časová pečiatka v čase odoslania dát – formát: “yyyy-MM-dd HH:mm:ss.SSS”
        self.velocity: float  # Rýchlosť vozidla
        self.real_time_computed: datetime
        self.rfid_tag: str  # RFID – nasnímaný RFID tag
        self.rssi_rfid: int  # RFID – sila signálu snímania RFID tagu
        self.msg_id: int  # Poradové číslo správy
        
    def real_start_time(self):
        return self.timestamp + timedelta(milliseconds=self.rel_time)
    
    @staticmethod
    def print_message_counts_per_unit(sensor_messages : List['SensorDataMessage']):
        counts = {}
        for msg in sensor_messages:
            uid = msg.unit_id
            if uid not in counts:
                counts[uid] = 0
            counts[uid] += 1

        for unit_id in counts:
            print(f"unit id: {unit_id}, počet správ: {counts[unit_id]}")

class SensorDataMessage(SmallSensorDataMessage):
    def __init__(self):
        
        self.container: str  # Nepoužíva sa
        
    
class SmallLitteringExecution:
    def __init__(self):
        self.id: int  # Db id
        self.unit_id: str  # Identifikátor zariadenia
        self.timestamp_start: datetime  # Časová pečiatka  - začiatok výsypu
        self.timestamp_end: datetime  # Časová pečiatka - koniec výsypu
        
        
        self.rfid_tag: str  # RFID – nasnímaný RFID tag
        self.car_arm: str  # Rameno vozidla na ktorom bol zaznamenaný výsyp – L/R
        
        self.is_paired_to_prediction = False
        
        self.is_original_prediction = True #false ak je umelo vytvorena rozdelenim B le
        self.is_delta_rfid : bool = False
        
        self.additional_info = ''
        self.trash_can = ''
        
        
        
        
class LitteringExecution(SmallLitteringExecution):
    def __init__(self):
        super().__init__()
        
        self.lat: float  # Zemepisná šírka
        self.lng: float  # Zemepisná dĺžka
        
        self.timestamp: datetime  # Časová pečiatka výsypu, formát: “yyyy-MM-dd HH:mm:ss”
        self.car_id: str  # Identifikátor vozidla
        self.een_status_h_id: int  # Nepoužíva sa
        self.provided: bool  # Príznak poskytnutia údajov na ďalšie spracovanie
        self.provided_at: str  # Časová pečiatka
        self.confirmed: bool  # Príznak potvrdzujúci prijatie údajov
        self.confirmed_at: str  # Časová pečiatka
        self.exception_type: str  # Nepoužíva sa
        self.weight: float  # Čistá hmotnosť odpadu
        self.enrich_weight: bool  # Príznak, požiadavka na doplnenie hmotnosti
        self.data_quality: str  # Final – python vyhodnotenie
        #toto bolo pridane
        self.trash_can : str
        
class VideoLitering:
    def __init__(self) -> None:
        self.id : int
        self.start : datetime
        self.end : datetime
        self.start_frame : int
        self.end_frame : int
        self.trash_can : str
        self.category : str
        self.additional_info : str


class WeightExecution:
    def __init__(self):
        self.id: int  # Db id
        self.unit_id: str  # Identifikátor zariadenia
        self.car_id: str  # EČV vozidla
        self.car_arm: str  # Rameno vozidla
        self.timestamp_start: datetime  # Začiatok merania
        self.timestamp_end: datetime  # Koniec merania
        self.weight: float  # Nameraná hmotnosť
        self.lat: float  # Zemepisná šírka
        self.lng: float  # Zemepisná dĺžka
        self.processed: bool  # Príznak spracovania