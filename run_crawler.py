from crawler.stocks_crawler import get_stocks, save_stock_codes_to_db
from crawler.posts_crawler import NaverCrawler
from crawler.new_price_crawler import get_last7_price_by_code
import os
import json
import sys
from server.app.db import get_db
from tqdm import tqdm
from tqdm.contrib import tmap
from tqdm.contrib.concurrent import process_map
from datetime import date, timedelta
import multiprocessing
from server.app.config import kospi_100

db = get_db()
stocks = db['StocksSample']
posts = db['Posts']

target_stocks = list(stocks.find())
print(target_stocks)

yesterday = date.today() - timedelta(1)
fr = dict(
    year=yesterday.year,
    month=yesterday.month,
    day=yesterday.day
)
print('crawling target day: ', fr)

MAX_PAGE = 1
# pc = Price_Crawler()


def crawl_price(crawl_list):
    print('start crawling...')
    for stock in crawl_list:
        result = get_last7_price_by_code(stock['code'])
        db['Stocks'].update_one({'_id': stock['_id']}, {
                                "$set": {'price': result}})


def crawl_posts(crawl_list):
    nc = NaverCrawler()

    print('start crawling...')
    for stock in crawl_list:
        res = nc.crawl(stock['code'], MAX_PAGE, fr)
        posts.insert_many(res)


if __name__ == '__main__':
    crawl_price(target_stocks)
