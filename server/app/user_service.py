from .db import get_db

db = get_db()

user1 = {
    'username': 'test user',
    'password': 'test pwd',
    'balance': 1000000,
    'stocks': [
        {'code': '066570', 'qty': 0},
        {'code': '006400', 'qty': 0},
        {'code': '000100', 'qty': 0},
    ]
}
