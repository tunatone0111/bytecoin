from .db import get_db
from .config import kospi_100

db = get_db()

def get_stock_list():
    return kospi_100

def get_current_stock_price(code):
    """ returns dict {'date': , 'price': } """
    stocks = db['StocksSample']
    return stocks.find_one({'code': code}, {'price': True})['price'][0]

def get_posts_by_stock_code(code):
    posts = db['Posts']
    return list(posts.find({'code': code}))
