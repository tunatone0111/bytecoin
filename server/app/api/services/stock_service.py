from ...db import get_db
from ...config import kospi_100
from pykrx import stock

db = get_db()
stocks = db['StocksWithLabel']
posts = db['Posts']

dto_projection = {'name': True, 'code': True, '_id': False, 'label': True, 'numPosts': True}


def read_all(filt={}):
    return list(stocks.find(filt, dto_projection))


def read_top5():
    return list(stocks.aggregate([{'$project': dto_projection}, {'$sort': {'numPosts': -1}}, {'$limit': 5}]))


def read_many(codes):
    return list(stocks.find({'code': {'$in': codes}}, dto_projection))


def read_one(code):
    return stocks.find_one({'code': code}, dto_projection)


def get_stock_list():
    return kospi_100


def get_current_stock_price(code):
    """ returns dict {'date': , 'price': } """
    return stocks.find_one({'code': code}, {'price': True})['price'][0]


def read_stock_with_posts(code):
    pipeline=[{
        '$match': {'code': code}
    },{
        '$lookup': {
            'from': 'Posts', 
            'let': {
                'c': '$code'
            }, 
            'pipeline': [{
                '$match': {
                    '$expr': {'$eq': ['$code', '$$c']}
                    }
                }, {
                    '$project': {'_id': False}
                }
            ],
            'as': 'posts'
        }
    },{
        '$project': {'_id': False}
    }
    ]
    return list(stocks.aggregate(pipeline))[0]

