
import sys # track error details
from .logger import logging #  import logger.py into exception.py

"""
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name[{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by zero error")
        raise CustomException(e, sys)
   
"""

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        self.error_message = self.error_message_detail(error_message, error_detail)

    def error_message_detail(self, error_message, error_detail):
        try:
            # Check if error_detail (traceback) exists
            if error_detail:
                exc_type, exc_value, exc_tb = error_detail.exc_type, error_detail.exc_value, error_detail.exc_tb
                file_name = exc_tb.tb_frame.f_code.co_filename
                line_number = exc_tb.tb_lineno
                return f"Error occurred in script {file_name} at line {line_number}. Error message: {str(exc_value)}"
            else:
                return f"Error: {str(error_message)}"
        except Exception as e:
            # If there's another problem, just show a general message
            return f"Error occurred: {str(e)}"
