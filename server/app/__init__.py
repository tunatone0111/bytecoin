from flask import Flask
from flask import render_template
from pymongo import MongoClient

from app.config import config_by_name


def create_app(config_name):
    app = Flask(__name__, static_url_path='',
                static_folder='../views', template_folder='../views')
    app.config.from_object(config_by_name[config_name])

    @app.route('/')
    def index():
        return render_template('index.html')

    return app
