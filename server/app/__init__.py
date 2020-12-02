from flask import Flask
from flask_cors import CORS

from .config import config_by_name, kospi_100

from .api.game import game
from .api.stocks import stocks
from .api.proxy import proxy
from .api.posts import posts


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])
    app.config['JSON_AS_ASCII'] = False

    app.register_blueprint(game, url_prefix='/api/game')
    app.register_blueprint(stocks, url_prefix='/api/stocks')
    app.register_blueprint(proxy, url_prefix='/api/proxy')
    app.register_blueprint(posts, url_prefix='/api/posts')

    CORS(app)

    return app
