from flask import Flask, render_template, url_for, request, redirect, Blueprint
from pymongo import MongoClient

from .config import config_by_name, kospi_100
from .db import get_db

from .api.game import game
from .api.stocks import stocks

from .api.services.user_service import read_users


def create_app(config_name):
    app = Flask(__name__,
                static_url_path='',
                static_folder='../views',
                template_folder='../views')
    app.config.from_object(config_by_name[config_name])
    app.config['JSON_AS_ASCII'] = False

    db = get_db()
    app.register_blueprint(game, url_prefix='/api/game')
    app.register_blueprint(stocks, url_prefix='/api/stocks')

    @app.route('/')
    def index():
        return 'index'

    @app.route("/game")
    def get_games():
        return 'game'

    @app.route('/users')
    def get_users():
        return {'users': read_users()}

    return app
