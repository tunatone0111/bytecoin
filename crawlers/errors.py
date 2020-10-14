class StockNotFoundException(Exception):
    def __init__(self, message):
        super(StockNotFoundException, self).__init__(message)

class DateNotInRangeException(Exception):
    pass