# system packages
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import json

# crawler packages
from mongodb import get_db
from errors import StockNotFoundException


def get_stocks():
    """Fetches the list of stock name and codes.

    Returns:
        stocks (list(dict)): list of stock object
    """
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
    codes = code_dataframes['code']
    names = code_dataframes['name']
    stocks = []
    for i in range(len(names)):
        stocks.append({
            'name': names[i],
            'code': codes[i]
        })
    return stocks


def save_stock_codes_to_db(stocks):
    """Saves stocks to MongoDB database

    Args:
        stocks (list(dict)): list of stocks
    """
    print('connecting to database...')
    Stocks = get_db()['Stocks']
    print('saving stocks to database')

    i = 0
    total = len(stocks)

    for stock in stocks:
        print(f'saving stock ({i+1}/{total}) : {stock["name"]}')
        i += 1

        found = Stocks.find_one(stock)  # search for matches for {name, code}
        if not found:
            Stocks.insert_one(stock)


def read_stock_codes_from_db():
    """Reads and returns stocks from MongoDB database

    Returns:
        list(dict): list of stocks
    """

    print('connecting to database...')
    Stocks = get_db()['Stocks']
    print('reading...')

    stocks = Stocks.find()
    return stocks


def get_stock_code(stock_name):
    """Get stock code by name

    Args:
        stock_name (string): stock name

    Returns:
        string: found stock code

    Raises:
        StockNotFoundException: returns when stock is not found.
    """

    Stocks = get_db()['Stocks']
    stock = Stocks.find({'name': stock_name})
    if stock:
        return stock['code']
    else:
        raise StockNotFoundException(
            f"Stock code is not found for stock name {stock_name}")


def get_stock_name(stock_code):
    """Get stock name by stock code

    Args:
        stock_code (string): stock code

    Raises:
        StockNotFoundException: returns when stock is not found

    Returns:
        string: found stock name
    """
    Stocks = get_db()['Stocks']
    stock = Stocks.find({'code': stock_code})
    if stock:
        return stock['code']
    else:
        raise StockNotFoundException(
            f'Stock name is not found for stock code {stock_code}')
