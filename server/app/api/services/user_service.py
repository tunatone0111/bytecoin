from bson.objectid import ObjectId
from ...db import get_db

db = get_db()
users = db['Users']

def read_users():
    result = list(users.find(projection={'_id': False}))
    return result


def read_user(user_id):
    result = users.find_one({'_id': ObjectId(user_id)})
    return result


def update_user(user_id, update_user_dto):
    return users.find_one_and_update({'_id': ObjectId(user_id)}, {'$set': update_user_dto})
