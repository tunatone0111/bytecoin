from ...config import kospi_100
from .user_service import read_user, update_user, read_users
from .stock_service import read_many, stocks as stocks_model
from ...errors import InvalidTradeException

import numpy as np
from random import randint
from pykrx import stock as krx_stock

TEST_PRICE = 10000

def reset_user_stocks(user_id="5fbd6ce41691156838210650"):
    num_stocks = 3

    stock_codes = []
    while len(stock_codes) != num_stocks:
        stock_name = np.array(kospi_100)[[randint(0, 99) for _ in range(num_stocks)]]
        stock_codes = list(stocks_model.find({'name': {'$in': stock_name.tolist()}}, {'_id': False, 'code': True}))
    
    y = randint(2000, 2020)
    m = randint(1, 12)
    s_date = f'{y}{str(m).zfill(2)}01'
    e_date = f"{y}{str(m).zfill(2)}{'28' if m == 2 else '30'}"
    
    prices = []
    for stock_code in stock_codes:
        res = krx_stock.get_market_ohlcv_by_date(s_date, e_date, stock_code['code'])['시가']
        prices.append(list(res))
    
    user = read_user(user_id)
    update_user(user_id, {
        'stocks': [{"price": price, "qty": 0} for price in prices],
        'curDate': 1,
        'cash': user['initialBalance']
    })    

    return True

def get_stocks_info(user_id="5fbd6ce41691156838210650"):
    user = read_user(user_id)
    stocks = read_many(list(user["stocks"].keys()))
    return {"stocks": stocks}


def _validation_arguments(f):
    def wrapper(stock_code, qty, user_id="5fbd6ce41691156838210650"):
        if qty < 0:
            raise InvalidTradeException('invalid qty')

        user = read_user(user_id)

        if stock_code not in user['stocks'].keys():
            raise InvalidTradeException('invalid stock id')

        return f(stock_code, qty, user_id, user)
    return wrapper


@_validation_arguments
def buy_stock(stock_code, qty, user_id, _user=None):
    cur_price = TEST_PRICE

    new_cash = _user['cash'] - cur_price * qty
    if new_cash < 0:
        raise InvalidTradeException('not enough money')

    new_qty = qty + _user['stocks'][stock_code]

    update_user(user_id, {'cash': new_cash,
                          f'stocks.{stock_code}': new_qty})

    return {"updated_cash": new_cash,
            "updated_stock_id": stock_code,
            "updated_qty": new_qty}


@_validation_arguments
def sell_stock(stock_code, qty, user_id, _user=None):
    new_qty = _user['stocks'][stock_code] - qty
    if new_qty < 0:
        raise InvalidTradeException('not enough qty')

    cur_price = TEST_PRICE

    new_cash = _user['cash'] + cur_price * qty

    update_user(user_id, {'cash': new_cash,
                          f'stocks.{stock_code}': new_qty})

    return {"updated_cash": new_cash,
            "updated_stock_id": stock_code,
            "updated_qty": new_qty}

def game_proceed(user_id="5fbd6ce41691156838210650"):
    user = read_user(user_id)
    new_date = user['curDate'] + 1
    if new_date > len(user['stocks'][0]['price']):
        return False
    update_user(user_id, {'curDate': new_date})
    return True
