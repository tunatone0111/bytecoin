from crawler.stocks_crawler import get_stocks, save_stock_codes_to_db
from crawler.posts_crawler import NaverCrawler
from crawler.price_crawler import NaverCrawler as Price_Crawler
import os
import json
import sys
from server.app.db import get_db
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from datetime import date, timedelta
import multiprocessing
from server.app.config import kospi_100

NSAMPLE = 20

db = get_db()
stocks = db['Stocks']

target_stocks = list(stocks.find({'name': {'$in': kospi_100}}))
print(target_stocks)

yesterday = date.today() - timedelta(1)
fr = dict(
    year=yesterday.year,
    month=yesterday.month,
    day=yesterday.day
)
print('crawling target day: ', fr)

MAX_PAGE = 1
pc = Price_Crawler()

crawl_list = list(target_stocks)
print(crawl_list)

def crawl_price():
    print('start crawling...')
    for stock in crawl_list:
        result = pc.crawl_page(stock=stock, multi_processed=True)
        cur_price = {"time": result['date'], "price": result['price']}
        if 'price' not in stock or stock['price'] is None:
            stock['price'] = [cur_price]
        else:
            stock['price'] = stock['price'][:99]
            stock['price'].insert(0, cur_price)
        stocks.update_one({'_id': stock['_id']}, {
                        "$set": {'price': stock['price']}})
        print(result)

def crawl_posts():
    nc = NaverCrawler()

    print('start crawling...')
    for stock in crawl_list:
        res = nc.crawl(stock['code'], MAX_PAGE, fr)
        print(res)

if __name__ == '__main__':
    pass