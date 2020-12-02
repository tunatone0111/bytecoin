from flask import Blueprint, jsonify, make_response, request

from ..config import kospi_100
from .services.stock_service import read_all, read_top5, read_one, read_many, stocks as stocks_model, read_stock_with_posts

stocks = Blueprint('stocks', __name__)

@stocks.route('/')
def get_all():
    result = read_all()
    assert(any(result))
    return jsonify(stocks=result)

@stocks.route('/search')
def search():
    q = request.args['q']
    pipeline=[
        {'$match': {'name': {'$regex': f'^{q}'}}},
        {'$sort': {'numPosts': 1}},
        {'$project': {'_id': False, "name": True, "code": True}}
    ]
    result = list(stocks_model.aggregate(pipeline))
    return jsonify(stocks=result)


@stocks.route('/top')
def get_top():
    result = read_top5()
    assert(any(result))
    return jsonify(stocks=result)


@stocks.route('/<code>')
def get_one(code):
    result = read_stock_with_posts(code)
    assert(result)
    return jsonify(stock=result)


@stocks.errorhandler(AssertionError)
def handle_None(e):
    print(e)
    return jsonify(stocks=None, errmsg='Not Found'), 404
