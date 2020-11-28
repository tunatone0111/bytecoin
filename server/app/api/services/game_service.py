from .user_service import read_user, update_user, read_users
from .stock_service import read_many
from ...errors import InvalidTradeException

TEST_PRICE = 10000


def get_stocks_info(user_id="5fbd6ce41691156838210650"):
    user = read_user(user_id)
    stocks = read_many(list(user["stocks"].keys()))
    return {"stocks": stocks}


def buy_stock(stock_code, qty, user_id="5fbd6ce41691156838210650"):
    if qty < 0:
        raise InvalidTradeException('invalid qty')

    user = read_user(user_id)

    if stock_code not in user['stocks'].keys():
        raise InvalidTradeException('invalid stock id')

    cur_price = TEST_PRICE

    new_balance = user['balance'] - cur_price * qty
    if new_balance < 0:
        raise InvalidTradeException('not enough money')

    new_qty = qty + user['stocks'][stock_code]

    update_user(user_id, {'balance': new_balance,
                          f'stocks.{stock_code}': new_qty})

    return {"updated_balance": new_balance,
            "updated_stock_id": stock_code,
            "updated_qty": new_qty}


def sell_stock(stock_code, qty, user_id="5fbd6ce41691156838210650"):
    if qty < 0:
        raise InvalidTradeException('invalid qty')

    user = read_user(user_id)

    if stock_code not in user['stocks']:
        raise InvalidTradeException('invalid stock id')

    new_qty = user['stocks'][stock_code] - qty
    if new_qty < 0:
        raise InvalidTradeException('not enough qty')

    cur_price = TEST_PRICE

    new_balance = user['balance'] + cur_price * qty

    update_user(user_id, {'balance': new_balance,
                          f'stocks.{stock_code}': new_qty})

    return {"updated_balance": new_balance,
            "updated_stock_id": stock_code,
            "updated_qty": new_qty}
