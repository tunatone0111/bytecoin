class StockNotFoundException(Exception):
    def __init__(self, message):
        super(StockNotFoundException, self).__init__(message)


class DateNotInRangeException(Exception):
    def __init__(self, message):
        super(DateNotInRangeException, self).__init__(message)


class HTMLElementNotFoundException(Exception):
    def __init__(self, message):
        super(HTMLElementNotFoundException, self).__init__(message)
