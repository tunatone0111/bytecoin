# system packages
from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent
from tqdm import tqdm

# crawler packages
from urltools import get_query
from stock_sources import NAVER
from errors import DateNotInRangeException, HTMLElementNotFoundException

# constants
TWO_DIGIT_TEMPLATE = "{0:0=2d}"


class Crawler():
    def __init__(self):
        # initialize fake user agent
        ua = UserAgent(verify_ssl=False)
        userAgent = ua.random
        self.headers = {'User-Agent': userAgent}
        self.result = list()


class NaverCrawler(Crawler):
    def template(self, stock_code, page):
        return f"https://finance.naver.com/item/board.nhn?code={stock_code}&page={page}"

    # crawls until it reaches the date or max pages.
    def crawl(self, stock_code, max_pages, date):
        self.result = []  # flush result array

        for page in range(1, 1 + max_pages):
            print(f"[page] ({page}/{max_pages})")
            done = self.crawl_page(stock_code, page, date)
            if done:
                break

        return self.result

    def crawl_page(self, stock_code, page, date):
        url = self.template(stock_code, page)
        html = requests.get(url, headers=self.headers).text
        soup = BeautifulSoup(html, 'html5lib')

        # find <a> tag elements whose link navigates to each post
        a_tag_elements = soup.select('#content td.title > a')

        # when we request page 100 on a stock that has only 50 pages, it returns the current page as '50'.
        # therefore, mismatch between requested page number, and the responded page means that
        # the crawler has reached the end page.
        if page != int(soup.select_one('td.on').get_text()):
            return True

        # get all post links.
        post_links = []
        for a_tag_element in a_tag_elements:
            post_links.append("https://finance.naver.com" +
                              a_tag_element['href'])

        # visit all post links and crawl them.
        for post_link in tqdm(post_links):
            try:
                post = self.crawl_post(post_link, date)
                # print(post)
            except DateNotInRangeException as e:
                print(e)
                break  # stop crawling when post date is earlier than the limit
            self.result.append(post)

    def crawl_post(self, post_link, date):
        try:
            html = requests.get(post_link, headers=self.headers).text
            soup = BeautifulSoup(html, 'html5lib')

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

            # filter dates
            year = date['year']
            month = TWO_DIGIT_TEMPLATE.format(date['month'])  # 1 => 01
            day = TWO_DIGIT_TEMPLATE.format(date['day'])  # 9 => 09
            compareable = f'{year}.{month}.{day}'

            # "2020.09.11" > "2020.09.10 12:30:15"
            # if the post date is earlier than the given date(comparable)
            if post_date < compareable:
                print(f'end. {post_date}',
                      compareable)
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

            return post

        except AttributeError as e:
            # catches if any HTML element does not exist.
            print(e)
            raise HTMLElementNotFoundException("Parsing Failed")
        

if __name__ == "__main__":
    # example code
    nc = NaverCrawler()

    fr = {
        'year': 2020,
        'month': 9,
        'day': 8
    }

    print('start crawling...')
    nc.crawl('005930', 1, fr)
