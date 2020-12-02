from flask import Blueprint, jsonify, make_response

proxy = Blueprint("proxy", __name__)

@proxy.route('/price/<stock_id>')
def get_price():
    '''Crawls Price Immediately'''
    return "Price"
