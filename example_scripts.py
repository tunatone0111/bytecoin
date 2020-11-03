from crawler.posts_crawler import NaverCrawler
from crawler.jsondb_services import insert_posts
from threading import Thread
import time


# 데이터베이스 업데이트하기
def update_posts_example():
    nc = NaverCrawler()

    fr = {
        'year': 2020,
        'month': 9,
        'day': 8
    }

    nc.crawl('005930', 1, fr)
    insert_posts(nc.result)


# 멀티스레딩
def work(stock_code, arr, fr, pages):
    nc = NaverCrawler(arr)
    nc.crawl(stock_code, pages, fr)


def multithreading_example(stock_codes, fr, max_pages):
    arr = []
    threads = [
        Thread(target=work, args=(stock_code, arr, fr, max_pages))
        for stock_code in stock_codes
    ]

    start = time.time()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    end = time.time()
    # print("[multithreading]", end - start, "seconds passed")
    return end - start


def singlethreaded_example(stock_codes, fr, max_pages):
    arr = []
    start = time.time()

    for stock_code in stock_codes:
        work(stock_code, arr, fr, max_pages)

    end = time.time()
    # print("[singlethreading]", end - start, "seconds passed")
    return end - start


def performance_test():
    stock_codes = [
        '005930',  # 삼성전자
        '020560',  # 아시아나항공
        '068270',  # 셀트리온
        '051910',  # 엘지화학
    ]

    fr = {
        'year': 2020,
        'month': 9,
        'day': 8
    }

    max_pages = 1

    print("Start Multithreading...")
    time.sleep(3)
    multithreading_time = multithreading_example(stock_codes, fr, max_pages)

    print("Start Singlethreading...")
    time.sleep(3)
    singlethreaded_time = singlethreaded_example(stock_codes, fr, max_pages)

    print(f"""[Performance Comparsion]
total stock count: {len(stock_codes)}
pages per each stock: {max_pages}

Multithreading: {multithreading_time} seconds passed.
Singlethreading: {singlethreaded_time} seconds passed.
""")
