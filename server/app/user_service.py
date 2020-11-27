from .db import get_db
from .stock_service import get_current_stock_price

db = get_db()

users = db['Users']


def get_user(id):
    return users.find_one({'_id': id})


def buy_stock(stock_code, qty, id="5fbd6ce41691156838210650"):
    if qty < 0:
        return False

    user = get_user(id)

    cur_price = int(get_current_stock_price(
        stock_code)['price'].replace(',', ''))

    new_balance = user['balance'] - cur_price * qty
    if new_balance < 0:
        return False

    new_qty = qty
    if stock_code in user['stocks']:
        new_qty = new_qty + user[stock_code]

    users.find_one_and_update(
        {'_id': id}, {'$set': {'balance': new_balance, f'stocks.{stock_code}': new_qty}})

    return True


def sell_stock(stock_code, qty, id="5fbd6ce41691156838210650"):
    if qty < 0:
        return False

    user = get_user(id)

    if stock_code not in user['stocks']:
        return False

    new_qty = user[stock_code] - qty
    if new_qty < 0:
        return False

    cur_price = int(get_current_stock_price(
        stock_code)['price'].replace(',', ''))

    new_balance = user['balance'] + cur_price * qty

    users.find_one_and_update(
        {'_id': id}, {'$set': {'balance': new_balance, f'stocks.{stock_code}': new_qty}})

    return True
