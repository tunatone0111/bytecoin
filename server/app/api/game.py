from flask import Blueprint, request, redirect, url_for, jsonify

from .services.game_service import buy_stock, sell_stock, get_stocks_info, reset_user_stocks, game_proceed
from ..errors import InvalidTradeException

game = Blueprint('game', __name__)
trade_ops = {'buy': buy_stock, 'sell': sell_stock}


@game.route('/prices')
def prices():
    return get_stocks_info()


@game.route('/reset')
def reset_game():
    return jsonify(res=reset_user_stocks())


@game.route('/<op>', methods=['POST'])
def trade(op):
    if op not in trade_ops.keys():
        return 'invalid operation', 400
    stock_code = request.form.get('stockId')
    count = int(request.form.get('count'))

    return trade_ops[op](stock_code, count)


@game.route('/proceed')
def proceed():
    return jsonify(data=game_proceed())


@game.errorhandler(InvalidTradeException)
def handle_bad_request(e):
    return jsonify(errmsg=str(e)), 400
