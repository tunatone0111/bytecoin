# system packages

from bs4 import BeautifulSoup
import numpy as np
import requests
import pandas as pd
from fake_useragent import UserAgent
import json
from multiprocessing import Process, Pool
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import csv
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import time
import datetime
import os
import argparse
import matplotlib.pyplot as plt
import random
# crawler packages
from ..urltools import get_query
from ..mongodb import get_db
from ..stock_sources import NAVER
from ..errors import DateNotInRangeException, HTMLElementNotFoundException
from ..stocks_crawler import get_stocks
from .posts_crawler_Hahversion1 import NaverCrawler
from .bert_classificaion_main import Bert_classification

# constants
how_old = 10
from .bert_config import KOSPI100


class multicrawl_and_return():
    def __init__(self, num_pages=1, days=0, hours=0, minutes=10, model_load=None, num_process=multiprocessing.cpu_count(), multiprocessing_flag=True, num_thread=5, MAX_LEN=65, epoch=1, batch_size=32, timed=False, when=None):

        if timed == True:
            self.date = datetime.datetime.now() - datetime.timedelta(days=days,
                                                                     hours=hours, minutes=minutes)
        else:
            self.date = when
        self.MAX_LEN = MAX_LEN
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_load = model_load

        self.num_process = num_process
        self.num_pages = num_pages
        self.num_thread = num_thread
        self.multiprocessing_flag = multiprocessing_flag

        db = get_db()
        coll = db['Stocks']
        self.stock_code_lst = list(coll.find({"name": {"$in": KOSPI100}}, {"_id": False, "name": True, "code": True}))

        self.crawl_lst = self.make_crawl_lst()

        self.model = NaverCrawler()

    def make_crawl_lst(self):
        temp_crawl_lst = []
        for i in range(len(self.stock_code_lst)):
            temp_crawl_lst.append(self.stock_code_lst[i]['code'])

        crawl_lst = []
        for i in temp_crawl_lst:
            crawl_lst += [[i, self.num_pages, self.date,
                           self.multiprocessing_flag, self.num_thread]]

        return crawl_lst

    def multi_crawl_and_filter2longsent(self):
        print('start crawling...')
        g = time.time()
        p = Pool(processes=self.num_process)
        p.starmap(self.model.crawl, self.crawl_lst[:])
        p.close()
        p.join()
        gr = time.time()
        print('lengh of nc.result is :       ', len(self.model.result))
        # for i in nc.result:
        # print('title is :',type(i['title']),'content is : ',type(i['content']))
        print(f'time spent when {self.num_process} proceesse and {self.num_thread} thread per process for '
              f'{len(self.stock_code_lst)} jobs and {self.num_pages} pages per job : {gr - g}')

        filtered_data = self.model.remove_2longsent()

        # flush result
        self.model.flush_result()

        self.model.result = filtered_data

        return self.model.result

    def content_extraction(self):
        contents_extracted_lst = []
        total_content_data = self.multi_crawl_and_filter2longsent()
        for id_x, i in enumerate(total_content_data):
            contents_extracted_lst.append(i['title'] + ' ' + i['content'])

        return contents_extracted_lst

    def get_label_lst(self):
        contents_extracted_lst = self.content_extraction()

        # BERT의 입력 형식에 맞게 변환
        sentences = [
            "[CLS] " + str(sentence) + " [SEP]" for sentence in contents_extracted_lst]
        # BERT의 토크나이저로 문장을 토큰으로 분리
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-multilingual-cased', do_lower_case=False)
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        print(sentences[0])
        print(tokenized_texts[0])
        # 토큰을 숫자 인덱스로 변환
        input_ids = [tokenizer.convert_tokens_to_ids(
            x) for x in tokenized_texts]

        # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
        input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN, dtype="long", truncating="post",
                                  padding="post")
        if self.model_load == None:
            bert = Bert_classification(
                epoch=self.epoch, batch_size=self.batch_size)
            print('----------------------------------------훈련되지 않은 모델을 사용합니다!-------------------------------------')
            print('----------------------------------------훈련되지 않은 모델을 사용합니다!-------------------------------------')
            print('----------------------------------------훈련되지 않은 모델을 사용합니다!-------------------------------------')
            print('----------------------------------------훈련되지 않은 모델을 사용합니다!-------------------------------------')
            print('----------------------------------------훈련되지 않은 모델을 사용합니다!-------------------------------------')
            label_lst = bert.work(input_ids)
        else:
            bert = Bert_classification(
                epoch=self.epoch, batch_size=self.batch_size, model_load_location=self.model_load)

            print('----------------------------------------훈련된 모델을 사용합니다--------------------------------------------')
            print('----------------------------------------훈련된 모델을 사용합니다--------------------------------------------')
            print('----------------------------------------훈련된 모델을 사용합니다--------------------------------------------')
            print('----------------------------------------훈련된 모델을 사용합니다--------------------------------------------')
            label_lst = bert.work(input_ids)

        return label_lst

    def combine_label_and_contents_lst(self):
        label_lst = self.get_label_lst()
        contents_lst = self.model.result
        for id_x, contents in enumerate(contents_lst):
            contents['label'] = label_lst[id_x]

        return contents_lst

    def save_result2db(self):
        contents_lst = self.combine_label_and_contents_lst()
        db = get_db()
        coll = db['Posts']
        coll.insert_many(contents_lst)

        self.model.flush_result()
        print('Saving files Completed!')
        doc = coll.find()
        for i in doc:
            print('is is : ', i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-M', '--modelload_loc', dest='model_load_loc', default=None, type=str, help='Location of model you want to load, format'
                        ': ~/home/abc123/model.tar')
    parser.add_argument('-t', '--threads', dest='threads',
                        default=5, type=int, help='쓰레드 개수를 입력하세요, 추천 : 5')
    parser.add_argument('-p', '--process', dest='process',
                        default=multiprocessing.cpu_count(), type=int, help='프로세스 개수를 입력하세요, 추천 : default ')
    parser.add_argument('-d', '--days', dest='days',
                        default=0, type=int, help='며칠 뒤로 가고 싶으신가요?')
    parser.add_argument('-H', '--hours', dest='hours',
                        default=0, type=int, help='몇시간 뒤로 가고싶으신가요?')
    parser.add_argument('-m', '--minutes', dest='minutes',
                        default=10, type=int, help='몇분 뒤로 가고 싶으신가요?')
    parser.add_argument('-b', '--batch_size', dest='batch_size',
                        default=4, type=int, help='batch_size number!')
    parser.add_argument('-w', '--way', dest='way', default='timed', type=str, choices=['timed', 'date'],
                        help='what do you want to do? train or test?')
    args = parser.parse_args()

    days = args.days
    hours = args.hours
    minutes = args.minutes
    batch_size = args.batch_size
    way = args.way
    num_thread = args.threads
    num_process = args.process
    model_load = args.model_load_loc

    if way == 'timed':
        multicrawl = multicrawl_and_return(batch_size=batch_size, days=days, hours=hours, minutes=minutes,
                                           timed=True, model_load=model_load, num_thread=num_thread, num_process=num_process)
        do_it = multicrawl.save_result2db()
        print(multicrawl.model.result)
    elif way == 'date':
        print('년월일시분 순으로 다음 포맷과 같이 입력하세요 : 2020-12-01-07-30')
        date = input()
        datetime_date = datetime.datetime.strptime(date, '%Y-%m-%d-%H-%M')
        multicrawl = multicrawl_and_return(batch_size=batch_size, days=days, hours=hours, minutes=minutes,
                                           timed=False, model_load=model_load, when=datetime_date, num_thread=num_thread, num_process=num_process)
        do_it = multicrawl.save_result2db()
        print(multicrawl.model.result)
    else:
        print('NOOOOOO! input is timed or date only!')
