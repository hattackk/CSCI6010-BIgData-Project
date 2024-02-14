import pytest
from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine.reflection import Inspector

from database_tables import Base

from dotenv import load_dotenv
import os
# load the .env file and set up variables
load_dotenv()
usr = os.environ.get('DB_USER')
pwd = os.environ.get("DB_PWD")
db_host = os.environ.get("DB_HOST")
db_port = os.environ.get("DB_PORT")
db_db = os.environ.get("DB_DATABASE")
DATABASE_URI = f'postgresql+psycopg2://{usr}:{pwd}@{db_host}:{db_port}/{db_db}'


@pytest.fixture(scope='module')
def engine():
    return create_engine(DATABASE_URI)

@pytest.fixture(scope='module')
def connection(engine):
    conn = engine.connect()
    yield conn
    conn.close()

@pytest.fixture(scope='module')
def setup_database(engine, connection):
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)