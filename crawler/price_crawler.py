# system packages
import datetime
import time
from bs4 import BeautifulSoup
import numpy as np
import requests
import pandas as pd
from fake_useragent import UserAgent
import json
from multiprocessing import Process, Pool
import threading
import multiprocessing

# crawler packages
from .urltools import get_query
from .mongodb import get_db
from .stock_sources import NAVER
from .errors import DateNotInRangeException, HTMLElementNotFoundException
import os
from .stocks_crawler import get_stocks
from concurrent.futures import ThreadPoolExecutor

# constants
TWO_DIGIT_TEMPLATE = "{0:0=2d}"


class Crawler():
    ua = UserAgent(verify_ssl=False)
    userAgent = ua.random
    headers = {'User-Agent': userAgent}

    def __init__(self):
        # initialize fake user agent
        # initialize result array
        self.result = multiprocessing.Manager().list()
        #self.result = []


class NaverCrawler(Crawler):

    def flush_result(self):
        # flush result manually
        self.result = multiprocessing.Manager().list()

    def crawl_page(self, stock, multi_processed):
        url = f"https://finance.naver.com/item/main.nhn?code={stock['code']}"
        html = requests.get(url, headers=self.headers).text
        soup = BeautifulSoup(html, 'html.parser')

        proc_id = os.getpid()

        # filter dates
        # year = date['year']
        # month = TWO_DIGIT_TEMPLATE.format(date['month'])  # 1 => 01
        # day = TWO_DIGIT_TEMPLATE.format(date['day'])  # 9 => 09
        # now_date = f'{year}.{month}.{day}'

        now_date = datetime.datetime.now()

        try:
            # find <no_today> tag elements
            no_today_elements = soup.find("p", {"class": "no_today"})
            blind_elements = no_today_elements.find("span", {"class": "blind"})
            now_price = blind_elements.text
        except:
            print('something is wrong here 1')

        post = {
            'code': stock['code'],  # stock code
            'name': stock['name'],
            'date': now_date,
            'price': now_price
        }

        print(f"price crawling of stock name : {post['name']}",
              f"  completed by process id : {proc_id} by {'using multiprocessing' if multi_processed else 'single processing'}")

        return post


'''
        except AttributeError as e:
            # catches if any HTML element does not exist.
            print(e)
            raise HTMLElementNotFoundException("Parsing Failed")
'''


if __name__ == "__main__":
    # example code
    nc = NaverCrawler()
    max_pool = 4
    num_cores = multiprocessing.cpu_count()
    print('number of cores :      ', num_cores)

    stock_code_list = nc.stock_code_list
    print(stock_code_list)

    crawl_list = []
    for i in range(len(stock_code_list)):
        crawl_list.append(stock_code_list[i]['code'])

    print('start crawling...')

    real_crawl_list = []
    for i in crawl_list:
        real_crawl_list += [[i, True]]

    real_crawl_list2 = []
    for i in crawl_list:
        real_crawl_list2 += [[i, False]]

    #a = time.time()
    # for i in range(len(nc.stock_code_list)):
    #    nc.crawl_page(real_crawl_list2[i][0],real_crawl_list2[i][1])
    #ar = time.time()
    #b = len(nc.result)
    # nc.flush_result()

    g = time.time()
    p = Pool()
    p.starmap(nc.crawl_page, real_crawl_list[:len(nc.stock_code_list)])
    p.close()
    p.join()
    gr = time.time()
    #print('lengh of nc.result is :       ', b)
    print('lengh of nc.result is :       ', len(nc.result))
    #print('time spent when single_processed : ', ar-a)
    print('time spent for 16 jobs : ', gr - g)
    for i in nc.result:
        print(i)


'''    
    #nc.crawl('005930', 2, fr)
    db = get_db()
    coll = db.stock
    real_result =[]
    real_result+=nc.result

    print(type(nc.result))
    coll.insert_many(real_result)
    print('---------------------------------------------------------------------------')
    print(nc.result)
    time.sleep(2)
    print('flushing result')
    nc.flush_result()
    time.sleep(2)
    print(nc.result)

    #nc.crawl('005930', 100, fr)
    #for i in nc.result:
        #print(i)

    #doc = db.collection.find({})
    #for i in doc:
        #print(i)
'''
