from mongodb_config import mongoconfig
import pymongo


def get_db():
    username = mongoconfig['username']
    password = mongoconfig['password']
    host = mongoconfig['host']
    port = mongoconfig['port']
    database = mongoconfig['database']
    authSource = mongoconfig['authSource']
    connection = pymongo.MongoClient(
            f"mongodb://{username}:{password}@{host}:{port}/{database}", authSource=authSource)
    db = connection[database]

    return db

if __name__=="__main__":
    db = get_db()
