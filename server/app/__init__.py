from flask import Flask
from flask import render_template
from pymongo import MongoClient

from .config import config_by_name
from .db import get_db


def create_app(config_name):
    app = Flask(__name__,
                static_url_path='',
                static_folder='../views',
                template_folder='../views')
    app.config.from_object(config_by_name[config_name])

    db = get_db()

    @app.route('/')
    def index():
        print(db['Stocks'])
        return render_template('index.html')

    return app
