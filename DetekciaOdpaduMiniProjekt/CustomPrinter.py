class CustomTextObj:
    def __init__(self) -> None:
        self.text = ''

class CustomPrinter:
    
    @staticmethod 
    def custom_print(text : str, should_print : bool = True, text_to_append  = None):
        if should_print:
            print(text)
        if text_to_append is not None:
            text_to_append.text =  text_to_append.text + f'\n{text}'
        
    