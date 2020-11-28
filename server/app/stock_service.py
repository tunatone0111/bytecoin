from .db import get_db
from .config import kospi_100

demo_flag = True
stocks_coll_name = 'StocksSample' if demo_flag else 'Stocks'

db = get_db()
stocks = db[stocks_coll_name]
posts = db['Posts']

dto_projection = {'name': True, 'code': True,
                  '_id': False, 'label': True}


def read_all():
    return list(stocks.find({}, dto_projection))


def read_top5():
    return list(stocks.aggregate([{'$project': dto_projection}, {'$sort': {'label': -1}}, {'$limit': 5}]))


def read_one(code):
    return list(stocks.find({'code': code}, dto_projection))


def get_stock_list():
    return kospi_100


def get_current_stock_price(code):
    """ returns dict {'date': , 'price': } """
    return stocks.find_one({'code': code}, {'price': True})['price'][0]


def get_posts_by_stock_code(code):
    return list(posts.find({'code': code}))
