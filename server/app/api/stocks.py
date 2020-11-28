from flask import Blueprint, jsonify, make_response

from ..config import kospi_100
from .services.stock_service import read_all, read_top5, read_one

stocks = Blueprint('stocks', __name__)


@stocks.route('/')
def get_all():
    result = read_all()
    assert(any(result))
    return jsonify(stocks=result)


@stocks.route('/top')
def get_top():
    result = read_top5()
    assert(any(result))
    return jsonify(stocks=result)


@stocks.route('/<code>')
def get_one(code):
    result = read_one(code)
    assert(any(result))
    return jsonify(stocks=result)


@stocks.errorhandler(AssertionError)
def handle_None(e):
    print(e)
    return jsonify(stocks=None, errmsg='Not Found'), 404
