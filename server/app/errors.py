class InvalidTradeException(Exception):
    def __init__(self, msg):
        super(InvalidTradeException, self).__init__(msg)
