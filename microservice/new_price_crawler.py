from urllib.request import urlopen
from bs4 import BeautifulSoup
from pandas import pandas as pd


def get_xml(code):
    url = f'http://asp1.krx.co.kr/servlet/krx.asp.XMLSise?code={code}'
    req = urlopen(url)
    result = req.read()
    return BeautifulSoup(result, 'lxml-xml')


def get_current_price_by_code(code):
    stock = get_xml(code).find('TBL_StockInfo')

    stock_df = pd.DataFrame(stock.attrs, index=[0])
    stock_df = stock_df.applymap(lambda x: x.replace(".", ""))

    return stock_df.to_dict()['CurJuka'][0]


def get_last7_price_by_code(code):
    stocks = get_xml(code).find_all('DailyStock')

    stocks_df = []
    for stock in stocks:
        stock_df = pd.DataFrame(stock.attrs, index=[0])
        stocks_df.append(stock_df.applymap(lambda x: x.replace(".", "")))

    return [{'date': x['day_Date'][0],
             'price': x['day_EndPrice'][0]}
            for x in stocks_df]


if __name__ == "__main__":
    print(get_last7_price_by_code('139480'))
