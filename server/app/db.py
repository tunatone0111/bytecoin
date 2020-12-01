from pymongo import MongoClient, errors
from .config import db_pwd


def get_db():
    if db_pwd == None:
        print("FATAL: Please set BYTECOIN_DB_PASSWORD env var")
        exit()

    try:
        connection = MongoClient(
            f"mongodb://aistartup:{db_pwd}@localhost:27017/bytecoin",
            authSource='admin')
        connection.server_info()
        db = connection['bytecoin']

        return db

    except errors.ServerSelectionTimeoutError as err:
        print(err)
        exit()
