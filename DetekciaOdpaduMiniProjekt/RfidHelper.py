class RFIDHelper:
    @staticmethod
    def is_empty_rfid(rfid : str) -> bool:
        result =  rfid == None or rfid == '' or rfid == 'NULL' or rfid == 'decoding_error'
        if not result and len(rfid) != 12:
            print(rfid)
        return result
    
    @staticmethod
    def is_big_container(rfid_tag : str):
        if len(rfid_tag) < 5:
            return False  # neplatný kód
        big_container_codes = {'E', 'J', 'K', 'Z'}
        return rfid_tag[4] in big_container_codes
    
    @staticmethod
    def replace_first_letter_with_x(rfid_tag : str) -> str:
        return "X" + rfid_tag[1:]
        