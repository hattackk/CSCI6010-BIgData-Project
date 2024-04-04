import os
import argparse

from sqlalchemy import Column, Integer, Float, String, MetaData, Table, text
from sqlalchemy import BigInteger, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine, select, update, exc

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

metadata = MetaData()

# putting this at the top since it wasn't in the original file
# this table will be used for tracking the status of the game review downloads
game_review_download_status_table = Table(
    'game_review_download_status', metadata,
    Column('game_id', Integer, primary_key=True),
    Column('status', String)
)

games_table = Table(
    'games', metadata,
    Column('game_id', Integer, primary_key=True),
    Column('game_name', String, nullable=False, unique=False),
    Column('developer', String),
    Column('publisher', String),
    Column('owners', String),
    Column('price', Float),
    Column('initialprice', Float),
    Column('discount', Float),
    Column('ccu', Integer)
)

game_rating_table = Table(
    'game_rating', metadata,
    Column('game_id', Integer, primary_key=True),
    Column('score_rank', Integer),
    Column('positive', Integer),
    Column('negative', Integer),
    Column('userscore', Integer),
    Column('average_forever', Integer),
    Column('average_2weeks', Integer),
    Column('median_forever', Integer),
    Column('median_2weeks', Integer)
)

game_review_summary_table = Table(
    'game_review_summary', metadata,
    Column('game_id', Integer, primary_key=True),
    Column('num_reviews', Integer),
    Column('review_score', Integer),
    Column('review_score_desc', String),
    Column('total_positive', Integer),
    Column('total_negative', Integer),
    Column('total_reviews', Integer)
)

game_reviews_table = Table(
    'game_reviews', metadata,
    Column('recommendationid', BigInteger, primary_key=True),
    Column('steamid', BigInteger),
    Column('language', String),
    Column('review', String),
    Column('timestamp_created', Integer),
    Column('timestamp_updated', Integer),
    Column('voted_up', Boolean),
    Column('votes_up', BigInteger),
    Column('votes_funny', BigInteger),
    Column('weighted_vote_score', Float),
    Column('comment_count', Integer),
    Column('steam_purchase', Boolean),
    Column('received_for_free', Boolean),
    Column('written_during_early_access', Boolean),
    Column('hidden_in_steam_china', Boolean),
    Column('steam_china_location', String),
    Column('application_id', Integer, ForeignKey('games.game_id')),
    Column('playtime_forever', Integer),
    Column('playtime_last_two_weeks', Integer),
    Column('playtime_at_review', Integer),
    Column('last_played', Integer),
    Column('is_spam', Integer),
    Column('sentiment_score', Float)
)

steam_users_table = Table(
    'steam_users', metadata,
    Column('steamid', BigInteger, primary_key=True),
    Column('num_games_owned', Integer),
    Column('num_reviews', Integer),
)

# database params
load_dotenv()
usr = os.environ.get('DB_USER')
pwd = os.environ.get("DB_PWD")
db_host = os.environ.get("DB_HOST")
db_port = os.environ.get("DB_PORT")
db_db = os.environ.get("DB_DATABASE")
DATABASE_URI = f'postgresql+psycopg2://{usr}:{pwd}@{db_host}:{db_port}/{db_db}'
print(DATABASE_URI)
engine = create_engine(DATABASE_URI)
metadata.create_all(engine)

def exec_statement(stmt):
    with engine.connect() as conn:
        result = conn.execute(stmt).all()
        if not result or len(result) == 0:
            print('Empty result.')
        conn.commit()

def get_row_count(table_name, conn):
    count_query = text(f"SELECT COUNT(*) FROM public.{table_name};")
    count_result = conn.execute(count_query).fetchone()
    return count_result[0]

def download_table_data(table, chunksize=10000):
    with engine.connect() as conn:
        row_count = get_row_count(table.name, conn)
        if row_count == 0:
            print(f"Table {table.name} has no data.")
            return pd.DataFrame()

        # Creating a progress bar
        with tqdm(total=row_count, desc=f"Downloading {table.name}") as pbar:
            # Initialize an empty DataFrame
            full_df = pd.DataFrame()
            for offset in range(0, row_count, chunksize):
                chunk_query = select(table).limit(chunksize).offset(offset)
                try:
                    chunk_df = pd.read_sql(chunk_query, conn)
                    full_df = pd.concat([full_df, chunk_df], ignore_index=True)
                    pbar.update(chunk_df.shape[0])
                except exc.SQLAlchemyError as e:
                    print(f"Error downloading data from table {table.name}: {e}")
                    break
            return full_df

def save_dataframe_as_pickle(df, table_name):
    if df is not None and not df.empty:
        filename = get_table_cache_location(table_name)
        df.to_pickle(filename)
        print(f"Saved {table_name} to {filename}")
    else:
        print(f"No data to save for {table_name}")

def get_table_cache_location(table_name):
    return f'./cache/${table_name}.pkl'

def is_table_cached(table_name):
    return os.path.exists(get_table_cache_location(table_name))

def main():
    parser = argparse.ArgumentParser(description="Download and cache database tables.")
    parser.add_argument("-o", "--override", action="store_true", help="Override existing cached tables.")
    args = parser.parse_args()

    # List of tables to process
    tables = [
        game_review_download_status_table,
        games_table,
        game_rating_table,
        game_review_summary_table,
        game_reviews_table,
        steam_users_table
    ]

    for table in tables:
        table_name = table.name
        if not args.override and is_table_cached(table_name):
            print(f"Table {table_name} is already cached. Skipping...")
            continue

        df = download_table_data(table)
        save_dataframe_as_pickle(df, table_name)

    print("All tables processed.")

if __name__ == '__main__':
    main()