from flask import Blueprint, jsonify, make_response, request

from ..config import kospi_100
from .services.stock_service import read_all, read_top5, read_one, read_many, stocks as stocks_model

 # 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

stocks = Blueprint('stocks', __name__)

@stocks.route('/')
def get_all():
    result = read_all()
    assert(any(result))
    return jsonify(stocks=result)

@stocks.route('/search')
def search():
    q = request.args['q']
    parsed_q = []
    for w in list(q.strip()):
        if '가'<=w<='힣':
            ch1=(ord(w)-ord('가')) // 588
            ch2=((ord(w)-ord('가')) - 588*ch1) // 28
            ch3=(ord(w)-ord('가')) - 588*ch1 - 28*ch2
            parsed_q.append([CHOSUNG_LIST[ch1], JUNGSUNG_LIST[ch2], JONGSUNG_LIST[ch3]])
        else:
            parsed_q.append(w)
    pipeline=[
        {'$match': {'name': {'$regex': f'^{q}'}}},
        {'$sort': {'numPosts': 1}},
        {'$project': {'_id': False, "name": True}}
    ]
    result = [x['name'] for x in stocks_model.aggregate(pipeline)]
    return jsonify(stocks=result)


@stocks.route('/top')
def get_top():
    result = read_top5()
    assert(any(result))
    return jsonify(stocks=result)


@stocks.route('/<code>')
def get_one(code):
    result = read_one(code)
    assert(result)
    return jsonify(stock=result)


@stocks.errorhandler(AssertionError)
def handle_None(e):
    print(e)
    return jsonify(stocks=None, errmsg='Not Found'), 404
