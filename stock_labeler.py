from server.app.db import get_db
from server.app.config import kospi_100
from random import uniform
db = get_db()

stocks = db['Stocks']

for i in kospi_100:
    updated_stock = stocks.find_one_and_update(
        {'name': {'$eq': i}},
        {'$set': {"label": uniform(0, 1)}}
    )
    print(updated_stock)
