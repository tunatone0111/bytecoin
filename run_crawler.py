from crawler.stocks_crawler import get_stocks, save_stock_codes_to_db
from crawler.posts_crawler import NaverCrawler
from crawler.price_crawler import NaverCrawler as Price_Crawler
import os
import json
from server.app.db import get_db
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from datetime import date, timedelta
import multiprocessing

NSAMPLE = 20

db = get_db()
stocks = db['Stocks']

kospi_100 = ["삼성전자", "SK하이닉스", "삼성바이오로직스", "NAVER", "현대차", "LG화학", "현대모비스", "셀트리온", "삼성물산", "LG생활건강", "POSCO", "신한지주", "삼성SDI", "KB금융", "SK텔레콤", "기아차", "SK", "한국전력", "삼성에스디에스", "삼성생명", "카카오", "엔씨소프트", "KT&G", "LG", "아모레퍼시픽", "SK이노베이션", "LG전자", "삼성화재", "하나금융지주", "삼성전기", "S-Oil", "한국조선해양", "넷마블", "우리금융지주", "고려아연", "롯데케미칼", "아모레G", "KT", "웅진코웨이", "기업은행", "강원랜드", "LG유플러스", "한온시스템", "현대글로비스", "LG디스플레이", "미래에셋대우", "현대중공업지주", "현대건설", "삼성중공업", "GS", "삼성카드",
             "오리온", "한국금융지주", "롯데지주", "현대제철", "한국타이어앤테크놀로지", "호텔신라", "CJ제일제당", "롯데쇼핑", "LG이노텍", "삼성엔지니어링", "에스원", "한미약품", "삼성증권", "NH투자증권", "이마트", "CJ대한통운", "한국가스공사", "두산밥캣", "DB손해보험", "한국항공우주", "GS리테일", "한화솔루션", "유한양행", "신세계", "BGF리테일", "대림산업", "휠라홀딩스", "대우조선해양", "CJ", "제일기획", "쌍용양회", "GS건설", "대한항공", "한미사이언스", "BNK금융지주", "금호석유", "하이트진로", "일진머티리얼즈", "팬오션", "포스코인터내셔널", "메리츠종금증권", "현대해상", "현대백화점", "SKC", "KCC", "대우건설", "오뚜기", "한화생명", "현대엘리베이터"]

target_stocks = list(stocks.find({'name': {'$in': kospi_100}}))
print(target_stocks)

yesterday = date.today() - timedelta(1)
fr = dict(
    year=yesterday.year,
    month=yesterday.month,
    day=yesterday.day
)
print('crawling target day: ', fr)


pc = Price_Crawler()

crawl_list = list(target_stocks)
print(crawl_list)

# print('start crawling...')
# for stock in crawl_list:
#     result = pc.crawl_page(stock=stock, multi_processed=True)
#     cur_price = {"time": result['date'], "price": result['price']}
#     if 'price' not in stock or stock['price'] is None:
#         stock['price'] = [cur_price]
#     else:
#         stock['price'] = stock['price'][:99]
#         stock['price'].insert(0, cur_price)
#     stocks.update_one({'_id': stock['_id']}, {
#                       "$set": {'price': stock['price']}})
#     print(result)


MAX_PAGE = 1
nc = NaverCrawler()

print('start crawling...')
for stock in crawl_list:
    res = nc.crawl(stock['code'], MAX_PAGE, fr)
    print(res)
