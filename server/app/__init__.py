from flask import Flask, render_template, url_for, request
from pymongo import MongoClient

from .config import config_by_name, kospi_100
from .db import get_db
from .stock_service import get_top5_stocks
import app.user_service


def create_app(config_name):
    app = Flask(__name__,
                static_url_path='',
                static_folder='../views',
                template_folder='../views')
    app.config.from_object(config_by_name[config_name])
    app.config['JSON_AS_ASCII'] = False

    db = get_db()

    @app.route('/')
    def index():
        stocks = get_top5_stocks()
        return render_template('index.html', stocks=stocks)

    @app.route("/game", methods=["GET"])
    def get_games():
        return render_template("game.html")

    def handle_trade(sevice_func):
        stock_code = request.form.get('stockId')
        count = int(request.form.get('count'))
        if not service_func(stock_code, count):
            print('failed')

    @app.route('/api/buy', methods=["POST"])
    def buy_stock():
        handle_trade(user_service.buy_stock)
        return redirect(url_for('/game'))

    @app.route('/api/sell', methods=["POST"])
    def sell_stock():
        handle_trade(user_service.sell_stock)
        return redirect(url_for('/game'))

    @app.route('/stocks')
    def get_stocks():
        result = list(db['Stocks'].find(
            {'name': {'$in': kospi_100}}, {'name': True, 'code': True, '_id': False, 'price': True}))
        return {"data": result}

    return app
