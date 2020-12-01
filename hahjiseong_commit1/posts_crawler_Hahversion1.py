# system packages
from bs4 import BeautifulSoup
import numpy as np
import requests
import pandas as pd
from fake_useragent import UserAgent
import json
from multiprocessing import Process,Pool
import threading
import multiprocessing
# crawler packages
from urltools import get_query
from mongodb import get_db
from stock_sources import NAVER
from errors import DateNotInRangeException, HTMLElementNotFoundException
import os
from stocks_crawler import get_stocks
from concurrent.futures import ThreadPoolExecutor
# constants
TWO_DIGIT_TEMPLATE = "{0:0=2d}"
import time
import csv

import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf

import pandas as pd
import numpy as np
import random
import time
import datetime
import os
import argparse
import matplotlib.pyplot as plt
import random

class Crawler():
    def __init__(self):
        # initialize fake user agent
        ua = UserAgent(verify_ssl=False)
        userAgent = ua.random
        self.headers = {'User-Agent': userAgent}
        # initialize result array
        self.result = multiprocessing.Manager().list()
        #self.result = []


class NaverCrawler(Crawler):
    def template(self, stock_code, page):
        return f"https://finance.naver.com/item/board.nhn?code={stock_code}&page={page}"

    def crawl(self, stock_code, max_pages, date,Multi_threading,num_worker):
        #Multi_threading 은 멀티스레딩을 할 것인지 여부 체크 용도
        #result = []  # flush result array
        proc_id = os.getpid()
        self.max_pages = max_pages


        if Multi_threading != True:
            for page in range(1, 1 + max_pages):
                print(f"[page] ({page}/{max_pages}) of stock {stock_code} by process id : {proc_id} and thread : {threading.current_thread()} when not mutithreaded")
                done = self.crawl_page(stock_code, page, date,threaded=False)

        elif Multi_threading == True:
            with ThreadPoolExecutor(max_workers=num_worker) as executor:
                future2execute =  {executor.submit(self.crawl_page,stock_code,page,date,True):page for page in range(1,1+max_pages)}

    def flush_result(self):
        #flush result manually
        self.result = multiprocessing.Manager().list()

    def crawl_page(self, stock_code, page, date,threaded):
        url = self.template(stock_code, page)
        html = requests.get(url, headers=self.headers).text
        soup = BeautifulSoup(html, 'html.parser')

        # find <a> tag elements whose link navigates to each post
        a_tag_elements = soup.select('#content td.title > a')

        # when we request page 100 on a stock that has only 50 pages, it returns the current page as '50'.
        # therefore, mismatch between requested page number, and the responded page means that
        # the crawler has reached the end page.
        if page != int(soup.select_one('td.on').get_text()):
            return

        # get all post links.
        post_links = []
        for a_tag_element in a_tag_elements:
            post_links.append("https://finance.naver.com" +
                              a_tag_element['href'])

        # visit all post links and crawl them.
        for post_link in post_links:
            try:
                post = self.crawl_post(post_link, date)
            except DateNotInRangeException as e:
                print(e)
                break  # stop crawling when post date is earlier than the limit
            self.result.append(post)

        proc_id = os.getpid()
        #threaded는 프린트문 실행 용도
        if threaded == True:
            print(f"[page] ({page}/{self.max_pages}) of stock {stock_code} by process id : {proc_id} and thread : {threading.current_thread()}")

    def crawl_post(self, post_link, date):
        try:
            html = requests.get(post_link, headers=self.headers).text
            soup = BeautifulSoup(html, 'html.parser')

            query_dict = get_query(post_link)

            # get metadata from querystring
            post_id = query_dict['nid'][0]
            stock_code = query_dict['code'][0]
            post_title = "undefined"

            # todo - specifiy title element in a more reliable way.
            title_element = soup.select_one("strong.c.p15")
            if title_element:
                post_title = title_element.get_text()

            post_content = soup.select_one("#body").get_text('\n', strip=True)
            post_date = soup.select_one(".gray03.p9.tah").get_text()
            post_date_str2datetime = datetime.datetime.strptime(post_date, '%Y.%m.%d %H:%M')

            # "2020.09.11" > "2020.09.10 12:30:15"
            # if the post date is earlier than the given date(comparable)
            if post_date_str2datetime < date:
                print(f'end bcause {post_date} is ealier than time set : {date}')
                # stop crawling for this post
                raise DateNotInRangeException("date is not in range")

            post_views = soup.select_one("span.tah.p11").get_text()
            post_good_count = soup.select_one("._goodCnt").get_text()
            post_bad_count = soup.select_one("._badCnt").get_text()

            # construct post object
            post = {
                'id': post_id,
                'source': NAVER,
                'code': stock_code,  # stock code
                'link': post_link,
                'title': post_title,
                'date': post_date,
                'views': post_views,
                'content': post_content,
                'good_count': post_good_count,
                'bad_count': post_bad_count,
            }
            #print(post)

            return post

        except AttributeError as e:
            # catches if any HTML element does not exist.
            print(e)
            raise HTMLElementNotFoundException("Parsing Failed")

    def percentile_check(self):
        temp_lst = []
        for id_x, i in enumerate(self.result):
            temp_lst.append(i['title'] + ' ' + i['content'])
            #print(temp_lst)
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        temp_txt = ["[CLS] " + str(sentence) + " [SEP]" for sentence in temp_lst]
        temp_tokenized_txt_lst = [tokenizer.tokenize(sentence) for sentence in temp_txt]
        temp_tokenized_txt_len_lst = [len(i) for i in temp_tokenized_txt_lst]
        #print(temp_tokenized_txt_len_lst)
        print(f'텍스트 수 : {len(temp_tokenized_txt_len_lst)} 에 대한 제 3분위값 = {np.percentile(temp_tokenized_txt_len_lst,75)}')
        plt.figure(figsize=(15,10))
        plt.hist(temp_tokenized_txt_len_lst,bins=100,range=[min(temp_tokenized_txt_len_lst)-10
                                                                ,100],density=True)
        plt.xlabel('lengths of sentences')
        plt.ylabel('probability')
        plt.show()

    def remove_2longsent(self):
        percentile_num = 65
        temp_contents_lst = []
        for id_x, i in enumerate(self.result):
            temp_contents_lst.append([i['title'] + ' ' + i['content'],i])

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        temp_txt_lst = ["[CLS] " + str(sentence) + " [SEP]" for sentence,other_content in temp_contents_lst]

        temp_tokenized_txt_lst = [[tokenizer.tokenize(sentence),len(tokenizer.tokenize(sentence))] for sentence in temp_txt_lst]
        mask = [txt[1]<65 for txt in temp_tokenized_txt_lst]

        contents_lst_shorter = [contents[1] for id_x,contents in enumerate(temp_contents_lst) if mask[id_x]]

        return contents_lst_shorter


if __name__ == "__main__":
    # example code
    nc = NaverCrawler()
    #max_pool = 4
    num_cores = multiprocessing.cpu_count()
    print('number of cores :      ',num_cores)

    stock_code_list = get_stocks()
    print(len(stock_code_list))


    fr = {
        'year': 2020,
        'month': 1,
        'day': 8
    }

    crawl_list = []
    for i in range(len(stock_code_list)):
        crawl_list.append(stock_code_list[i]['code'])

    print('start crawling...')

    real_crawl_list = []
    for i in crawl_list:
        real_crawl_list += [[i, 20, fr, True, 5]]

    g = time.time()
    p = Pool()
    p.starmap(nc.crawl, real_crawl_list[:])
    p.close()
    p.join()
    gr = time.time()
    print('lengh of nc.result is :       ', len(nc.result))
    #for i in nc.result:
        #print('title is :',type(i['title']),'content is : ',type(i['content']))
    print('time spent when 8 proceesse and 5 thread per process for 16 jobs and 10 pages per job : ', gr - g)

    test_lst = nc.remove_2longsent()
    #random.shuffle(test_lst)
    #test_lst2 = []
    #num_limit = 1000
    #for idx,i in enumerate(test_lst):
    #    if idx<num_limit:
    #        print(i)
    #datas = {'text':test_lst[:num_limit]}
    #dataframe = pd.DataFrame(datas,columns=['text'])
    #dataframe.to_csv("/home/awefjio12345/Downloads/crawl_txt_list/test.csv",header=False,index=False,encoding='utf-8-sig')









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