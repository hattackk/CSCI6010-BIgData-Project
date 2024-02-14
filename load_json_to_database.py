import json
import os

from dotenv import load_dotenv
import click
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import create_engine
from database_tables import games_table, Base, game_review_summary_table, game_rating_table
from sqlalchemy.inspection import inspect

# load the .env file and set up variables
load_dotenv()
usr = os.environ.get('DB_USER')
pwd = os.environ.get("DB_PWD")
db_host = os.environ.get("DB_HOST")
db_port = os.environ.get("DB_PORT")
db_db = os.environ.get("DB_DATABASE")
DATABASE_URI = f'postgresql+psycopg2://{usr}:{pwd}@{db_host}:{db_port}/{db_db}'


def process_game_file(games_file):
    # load the games_file
    with open(games_file, 'r') as f:
        games_file = json.loads(f.read())

    # format the game data to match the game table
    games = []
    games_review_summary = []
    game_rating = []
    for key, value in games_file.items():

        # Here is what is needed for the games table
        games.append(
            dict(
                game_id=value.get('appid'),
                game_name=value.get('name'),
                developer=value.get('developer'),
                publisher=value.get('publisher'),
                owners=value.get('owners'),
                price=value.get('price'),
                initialprice=value.get('initialprice'),
                discount=value.get('discount'),
                ccu=value.get('ccu'),
            )
        )
        # here is what is need for the game_review_summary table
        query_summary = value.get('query_summary')
        games_review_summary.append(
            dict(
                game_id=value.get('appid'),
                num_reviews=query_summary.get('num_reviews'),
                review_score=query_summary.get('review_score'),
                review_score_desc=query_summary.get('review_score_desc'),
                total_positive=query_summary.get('total_positive'),
                total_negative=query_summary.get('total_negative'),
                total_reviews=query_summary.get('total_reviews')
            )
        )
        # here is what is needed for the game_rating table
        game_rating.append(
            dict(
                game_id=value.get('appid'),
                score_rank=value.get('score_rank') if len(value.get('score_rank')) > 0 else 0,
                positive=value.get('positive'),
                negative=value.get('negative'),
                userscore=value.get('userscore'),
                average_forever=value.get('average_forever'),
                average_2weeks=value.get('average_2weeks'),
                median_forever=value.get('median_forever'),
                median_2weeks=value.get('median_2weeks')
            )
        )

    # set up the sql alchemy connection
    engine = create_engine(DATABASE_URI)
    Base.metadata.create_all(engine)

    with engine.connect() as conn:

        # TODO: (MJPM, 02/14/24) break into function, remove duplicate code
        for record in games:
            primary_keys = [key.name for key in inspect(games_table).primary_key]
            stmt = insert(games_table).values(record)
            # define dict of non-primary keys for updating
            update_dict = {
                c.name: c
                for c in stmt.excluded
                if not c.primary_key
            }
            update_stmt = stmt.on_conflict_do_update(
                index_elements=primary_keys,
                set_=update_dict,
            )

            result = conn.execute(update_stmt)

        for record in games_review_summary:
            primary_keys = [key.name for key in inspect(game_review_summary_table).primary_key]
            stmt = insert(game_review_summary_table).values(record)
            # define dict of non-primary keys for updating
            update_dict = {
                c.name: c
                for c in stmt.excluded
                if not c.primary_key
            }
            update_stmt = stmt.on_conflict_do_update(
                index_elements=primary_keys,
                set_=update_dict,
            )

            result = conn.execute(update_stmt)

        for record in game_rating:
            primary_keys = [key.name for key in inspect(game_rating_table).primary_key]
            stmt = insert(game_rating_table).values(record)
            # define dict of non-primary keys for updating
            update_dict = {
                c.name: c
                for c in stmt.excluded
                if not c.primary_key
            }
            update_stmt = stmt.on_conflict_do_update(
                index_elements=primary_keys,
                set_=update_dict,
            )

            result = conn.execute(update_stmt)

        conn.commit()


def process_review_file(review_file):
    pass


@click.command()
@click.option('--games_file', type=click.Path(exists=True), default=None, help='Path to games file')
@click.option('--review_file', type=click.Path(exists=True), default=None, help='Path to review file')
def main(games_file, review_file):
    # Connect to the database
    engine = create_engine(DATABASE_URI)
    Base.metadata.create_all(engine)
    if not games_file:
        games_file = 'top100in2weeks.json'
    # Process the files
    process_game_file(games_file)

    if not review_file:
        review_file = 'app_reviews_top_100_2weeks.json'
    process_review_file(review_file)


if __name__ == '__main__':
    main()


