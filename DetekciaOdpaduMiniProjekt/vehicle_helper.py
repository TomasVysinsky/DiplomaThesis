from typing import List


class VehicleHelper:
    @staticmethod
    def get_decoded_name(ecv : str) -> str:
        #klasicke auta
        if ecv == 'ZA127IR':
            return '001'
        elif ecv == 'ZA234JG':
            return '002'
        elif ecv == 'ZA255KC':
            return '003'
        elif ecv == 'ZA346KA':
            return '004'
        elif ecv == 'ZA499JN':
            return '005'
        #nove auta
        elif ecv == 'AA619SP':
            return '006'
        elif ecv == 'AA713KN':
            return '007'
        elif ecv == 'BT155HT':
            return '008'
        elif ecv == 'BT752GY':
            return '009'
        elif ecv == 'BT769GY':
            return '010'
        elif ecv == 'BT874HR':
            return '011'
        #auta s ecv
        elif ecv == 'ZA196JN_video':
            return '101'
        elif ecv == 'ZA234JG_video':
            return '102'
        elif ecv == 'ZA503JU_video':
            return '103'
        return ecv
        raise Exception(f'ecv: {ecv} neexistuje')
    
    @staticmethod
    def get_car_arm(ecv : str, unit_id: str) -> str:
        if unit_id == 'all':
            return 'all'
        if ecv == 'ZA127IR':
            if unit_id == '003300363136510A38383630':
                return "L"
            elif unit_id == '002B00224230501820313335':
                return 'R'
        elif ecv == 'ZA234JG':
            if unit_id == '004500483138511230343432':
                return "L"
            elif unit_id == '0029004B3433511230353832':
                return 'R'
        elif ecv == 'ZA255KC':
            if unit_id == '002900373433511230353832':
                return "L"
            elif unit_id == '002C00144230501820313335':
                return 'R'
        elif ecv == 'ZA346KA':
            if unit_id == '002B001C3433511230353832':
                return "L"
            elif unit_id == '0029003B3433511230353832':
                return 'R'
        elif ecv == 'ZA499JN':
            if unit_id == '0026002D3433511230353832':
                return "L"
            elif unit_id == '002B00444230501820313335':
                return 'R'
        #nove vozidla
        elif ecv == 'AA619SP':
            if unit_id == '002A00213433511230353832':
                return "L"
            elif unit_id == '002900483433511230353832':
                return 'R'
        elif ecv == 'AA713KN':
            if unit_id == '002900543433511230353832':
                return "L"
            elif unit_id == '002A00293433511230353832':
                return 'R'
        elif ecv == 'BT155HT':
            if unit_id == '002800353433511230353832':
                return "L"
            elif unit_id == '002B001E3433511230353832':
                return 'R'
        elif ecv == 'BT874HR':
            if unit_id == '0026001F3433511230353832':
                return "L"
            elif unit_id == '0028003E3433511230353832':
                return 'R'
        elif ecv == 'BT752GY':
            if unit_id == '004000434B31500720323957':
                return "L"
            elif unit_id == '002700573433511230353832':
                return 'R'
        elif ecv == 'BT769GY':
            if unit_id == '002C004B4230501820313335':
                return "L"
            elif unit_id == '002D00243136510A38383630':
                return 'R'
        
        #auta s ecv
        elif ecv == 'ZA196JN_video' or ecv == 'ZA196JN':
            if unit_id == '003A002C3038510E39363731':
                return "L"
            elif unit_id == '005800323138511836323738':
                return 'R'
        elif ecv == 'ZA234JG_video':
            if unit_id == '004500483138511230343432':
                return "L"
            elif unit_id == '0029004B3433511230353832':
                return 'R'
        elif ecv == 'ZA503JU_video' or ecv == 'ZA503JU':
            if unit_id == '002B000A4230501820313335':
                return "L"
            elif unit_id == '003F00553136510A38383630':
                return 'R'
        raise Exception(f'ecv: {ecv} unit id: {unit_id} neexistuje')
    
    @staticmethod
    def get_left_right_car_arm(ecv : str, unit_ids : List[str]):
        arm_first = VehicleHelper.get_car_arm(ecv, unit_ids[0])
        arm_second = VehicleHelper.get_car_arm(ecv, unit_ids[1])
        result = []
        if arm_first == 'L' and arm_second == 'R':
            result.append(unit_ids[0])
            result.append(unit_ids[1])
        elif arm_first == 'R' and arm_second == "L":
            result.append(unit_ids[1])
            result.append(unit_ids[0])
        return result
    
    @staticmethod
    def is_video_vehicle(ecv : str) -> bool:
        return 'video' in ecv