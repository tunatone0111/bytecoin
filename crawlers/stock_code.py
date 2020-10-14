from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import json
from mongodb import get_db


class StockNotFoundError:
    def __str__(self, name):
        return f"Stock name {name} is invalid"


def get_stock_codes_map():
    print("fetching remote...")
    code_dataframes = pd.read_html(
        'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0]
    # 우리가 필요한 것은 회사명과 종목코드이기 때문에 필요없는 column들은 제외해준다.
    print("parsing and filtering data...")
    code_dataframes.종목코드 = code_dataframes.종목코드.map('{:06d}'.format)
    # 한글로된 컬럼명을 영어로 바꿔준다.
    code_dataframes = code_dataframes[['회사명', '종목코드']]
    code_dataframes = code_dataframes.rename(
        columns={'회사명': 'name', '종목코드': 'code'})
    code = code_dataframes['code']
    name = code_dataframes['name']
    stock_codes = dict()
    for i in range(len(name)):
        stock_codes[name[i]] = code[i]
    return stock_codes

"""Saves Stock Codes to MongoDB"""
def save_stock_codes_to_db(stock_codes):
    print('connecting to database...')
    StockCodes = get_db()['StockCodes']
    print('saving stock codes to database')
    i = 0
    total = len(stock_codes)
    for name in stock_codes:
        print(f'saving stock ({i+1}/{total}) : {name}')
        i += 1
        code = stock_codes[name]
        found = StockCodes.find_one({"name": name, "code": code})
        if not found:
            StockCodes.insert_one({"name": name, "code": code})

"""Returns Stock Codes from Mongodb"""
def read_stock_codes_from_db():
    print('connecting to database...')
    StockCodes = get_db()['StockCodes']
    print('reading...')
    stock_codes_cursor = StockCodes.find()
    stock_codes = dict()
    for stock in stock_codes_cursor:
        stock_codes[stock['name']] = stock['code']
    print("read done")
    return stock_codes

"""Gets stock code by stock name"""
def get_stock_code(stock_name):
    stock_codes = read_stock_codes_from_db()
    try:
        stock_code = stock_codes[stock_name]
        return stock_code
    except KeyError:
        raise StockNotFoundError(stock_name)


if __name__ == "__main__":
    save_stock_codes_to_db(get_stock_codes_map())
