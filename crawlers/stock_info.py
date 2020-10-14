from bs4 import BeautifulSoup
import numpy as np
import requests
import pandas as pd
from fake_useragent import UserAgent
import json
from abc import *
from pprint import pprint
from stock_code import read_stock_codes_from_db, get_stock_code
from urltools import qdic
from mongodb import get_db
from multiprocessing import Process


class DateNotInRangeException(Exception):
    pass


class Crawler():
    def __init__(self):
        ua = UserAgent(verify_ssl=False)
        userAgent = ua.random
        self.headers = {'User-Agent': userAgent}
        self.result = []


class NaverCrawler(Crawler):
    def template(self, stock_code, page):
        return f"https://finance.naver.com/item/board.nhn?code={stock_code}&page={page}"

    def crawl_from_page(self, stock_code, max_pages, syncDB=False):
        self.result = []
        for page in range(1, 1 + max_pages):
            print(f"[page] ({page}/{max_pages})")
            self.crawl_page(stock_code, page, syncDB)

    def crawl_from_date(self, stock_code, max_pages, date, syncDB=False):
        self.result = []
        for page in range(1, 1 + max_pages):
            print(f"[page] ({page}/{max_pages})")
            done = self.crawl_page(stock_code, page, syncDB, True, date)
            if done:
                break

    def crawl_page(self, stock_code, page, syncDB=False, date_filter=False, date={}):
        url = self.template(stock_code, page)
        html = requests.get(url, headers=self.headers).text
        soup = BeautifulSoup(html, 'html5lib')

        # find <a> tag elements whose link navigates to each post
        a_tag_elements = soup.select(
            '#content td.title > a')
        if page != int(soup.select_one('td.on').get_text()):
            return True

        post_links = []
        for a_tag_element in a_tag_elements:
            post_links.append("https://finance.naver.com" +
                              a_tag_element['href'])
        for post_link in post_links:
            try:
                post = self.crawl_post(
                    post_link, date, date_filter)
            except DateNotInRangeException as e:
                print(e)
                break
            self.result.append(post)
            if syncDB:
                Posts = get_db()['Posts']
                if not Posts.find_one({'id': post['id']}):
                    Posts.insert_one(post)

    def crawl_post(self, post_link, date={}, date_filter=False):
        try:
            html = requests.get(post_link, headers=self.headers).text
            soup = BeautifulSoup(html, 'html5lib')

            qmap = qdic(post_link)
            post_id = qmap['nid'][0]
            stock_code = qmap['code'][0]
            post_page = int(qmap['page'][0])
            post_title = "undefined"
            titleel = soup.select_one("strong.c.p15")
            if titleel:
                post_title = titleel.get_text()
            print(f'[post] {post_title}')
            post_content = soup.select_one("#body").get_text('\n', strip=True)
            post_date = soup.select_one(".gray03.p9.tah").get_text()
            if date_filter:
                year = date['year']
                month = date['month']
                day = date['day']
                compareable = f'{year}.{"{0:0=2d}".format(month)}.{"{0:0=2d}".format(day)}'
                if post_date < compareable:
                    print(f'end. {post_date}',
                          compareable)
                    raise DateNotInRangeException()
            post_views = soup.select_one("span.tah.p11").get_text()
            post_good_count = soup.select_one("._goodCnt").get_text()
            post_bad_count = soup.select_one("._badCnt").get_text()

            post = {
                'id': post_id,
                'code': stock_code,  # stock code
                'link': post_link,
                'title': post_title,
                'date': post_date,
                'views': post_views,
                'content': post_content,
                'good_count': post_good_count,
                'bad_count': post_bad_count,
            }
            return post
        except AttributeError as e:
            print(e)
            raise Exception("Parsing Failed")


nc = NaverCrawler()

fr = {
    'year': 2020,
    'month': 10,
    'day': 8
}

p1 = Process(target=nc.crawl_from_date, args=('208850', 10, fr, True))
p2 = Process(target=nc.crawl_from_date, args=('223220', 10, fr, True))
p3 = Process(target=nc.crawl_from_date, args=('068270', 10, fr, True))

p1.start()
p2.start()
p3.start()

p1.join()
p2.join()
p3.join()
