from flask import Flask

from .config import config_by_name, kospi_100

from .api.game import game
from .api.stocks import stocks


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])
    app.config['JSON_AS_ASCII'] = False

    app.register_blueprint(game, url_prefix='/api/game')
    app.register_blueprint(stocks, url_prefix='/api/stocks')

    return app
