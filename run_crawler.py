from crawler.stocks_crawler import get_stocks, save_stock_codes_to_db
from crawler.posts_crawler import NaverCrawler
import os
import json
from server.app.db import get_db
from tqdm import tqdm

db = get_db()
stocks = db['Stocks']
stock_names = ['삼성전자', '셀트리온']
stock_codes = [stocks.find_one({'name': name})['code'] for name in stock_names]

fr = {
    'year': 2020,
    'month': 11,
    'day': 1
}

nc = NaverCrawler()

print('start crawling...')
for stock_code in tqdm(stock_codes):
    nc.crawl(stock_code, 1, fr)

# print(json.dumps(nc.result, indent=2, ensure_ascii=False))
