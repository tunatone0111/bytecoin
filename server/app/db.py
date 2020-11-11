from pymongo import MongoClient
from .config import db_pwd


def get_db():
    if db_pwd == None:
        print("FATAL: Please set BYTECOIN_DB_PASSWORD env var")
        exit()

    connection = MongoClient(
        f"mongodb://aistartup:{db_pwd}@localhost:27017/bytecoin",
        authSource='admin')

    db = connection['bytecoin']
    return db
