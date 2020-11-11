import os


class Config:
    SECRET_KEY = os.getenv('BYTECOIN_SECRET_KEY', 'bytecoinsecretkey')
    DB_PWD = os.getenv('BYTECOIN_DB_PASSWORD')
    DEBUG = False


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    PRESERVE_CONTEXT_ON_EXCEPTION = False


class ProductionConfig(Config):
    DEBUG = False


config_by_name = dict(
    dev=DevelopmentConfig,
    test=TestingConfig,
    prod=ProductionConfig
)

key = Config.SECRET_KEY
db_pwd = Config.DB_PWD
