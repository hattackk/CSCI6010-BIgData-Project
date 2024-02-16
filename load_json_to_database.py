import json
import os
import logging
from typing import List, Dict, Tuple

from dotenv import load_dotenv
import click
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import create_engine
from database_tables import (
    Table, metadata,
    games_table, game_review_summary_table, game_rating_table,
    steam_users_table, game_reviews_table
)


from sqlalchemy.inspection import inspect

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# load the .env file and set up variables
load_dotenv()
usr = os.environ.get('DB_USER')
pwd = os.environ.get("DB_PWD")
db_host = os.environ.get("DB_HOST")
db_port = os.environ.get("DB_PORT")
db_db = os.environ.get("DB_DATABASE")
DATABASE_URI = f'postgresql+psycopg2://{usr}:{pwd}@{db_host}:{db_port}/{db_db}'


def add_or_update(record_list: List[Dict], table: Table, conn):
    """
    Inserts new records or updates existing ones in the specified table.

    This function iterates through a list of dictionaries, each representing a record to be added or updated in the
    specified table. If a record with the same primary key(s) as the new record already exists in the table, the
    existing record is updated with the new values. Otherwise, a new record is inserted.

    Args:
        record_list (List[Dict]): A list of dictionaries, where each dictionary contains the data for a record to be
                                  inserted or updated.
        table (Table): The SQLAlchemy Table object representing the table to which the records will be inserted or
                       updated.
        conn: The SQLAlchemy connection object used to execute the insert or update statements.

    Returns:
        None
    """
    for record in record_list:
        primary_keys = [key.name for key in inspect(table).primary_key]
        stmt = insert(table).values(record)
        
        update_dict = {c.name: stmt.excluded[c.name] for c in table.columns if c.name not in primary_keys}
        
        update_stmt = stmt.on_conflict_do_update(
            index_elements=primary_keys,
            set_=update_dict,
        )
        
        result = conn.execute(update_stmt)
        logging.debug(f"Executed upsert for table {table.name}, affected rows: {result.rowcount}")

    conn.commit()


def parse_game_json(game_json) -> Tuple[List, List, List]:
    """
    :param game_json: dictionary of game_json with appid as the key
    :return: games, games_review_summary, game_rating each as a list
    """
    games = []
    games_review_summary = []
    game_rating = []
    for key, value in game_json.items():

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
        query_summary = value.get('query_summary', {})
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
                score_rank=value.get('score_rank') if isinstance(value.get('score_rank'), int) else 0,
                positive=value.get('positive'),
                negative=value.get('negative'),
                userscore=value.get('userscore'),
                average_forever=value.get('average_forever'),
                average_2weeks=value.get('average_2weeks'),
                median_forever=value.get('median_forever'),
                median_2weeks=value.get('median_2weeks')
            )
        )

    return games, games_review_summary, game_rating


def process_game_file(games_file: str, conn) -> None:
    """
    Processes a game file by loading its JSON content and updating the database tables.

    This function opens a JSON file containing game information, parses it to extract
    necessary data for three database tables (games, game_review_summary, and game_rating),
    and then updates these tables in the database using the provided connection.

    Args:
        games_file (str): The path to the JSON file containing game information.
        conn: The database connection object used to execute database operations.

    Returns:
        None
    """
    with open(games_file, 'r') as f:
        games_file_content = json.loads(f.read())

    games, games_review_summary, game_rating = parse_game_json(games_file_content) 
    add_or_update(games, games_table, conn)
    add_or_update(game_rating, game_rating_table, conn)
    add_or_update(games_review_summary, game_review_summary_table, conn)


def parse_single_game_review(json_obj) -> Tuple[Dict, Dict]:
    user = json_obj.get('author')

    single_review = dict(
            recommendationid=json_obj.get('recommendationid'),
            author=user.get('steamid'),
            language=json_obj.get('language'),
            review=json_obj.get('review'),
            timestamp_created=json_obj.get('timestamp_created'),
            timestamp_updated=json_obj.get('timestamp_updated'),
            voted_up=json_obj.get('voted_up'),
            votes_up=json_obj.get('votes_up'),
            votes_funny=json_obj.get('votes_funny'),
            weighted_vote_score=json_obj.get('weighted_vote_score'),
            comment_count=json_obj.get('comment_count'),
            steam_purchase=json_obj.get('steam_purchase'),
            received_for_free=json_obj.get('received_for_free'),
            written_during_early_access=json_obj.get('written_during_early_access'),
            hidden_in_steam_china=json_obj.get('hidden_in_steam_china'),
            steam_china_location=json_obj.get('steam_china_location'),
            application_id=json_obj.get('application_id')
        )
    single_user = dict(
            steamid=user.get('steamid'),
            num_games_owned=user.get('num_games_owned'),
            num_reviews=user.get('num_reviews'),
            playtime_forever=user.get('playtime_forever'),
            playtime_last_two_weeks=user.get('playtime_last_two_weeks'),
            playtime_at_review=user.get('playtime_at_review'),
            last_played=user.get('last_played'),
        )

    return single_review, single_user


def process_review_file(review_file, conn):
    game_reviews = []
    steam_users = []
    with open(review_file, 'r') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from line: {line}")
                continue
            single_review, single_user = parse_single_game_review(json_obj)
            game_reviews.append(single_review)
            steam_users.append(single_user)
    add_or_update(steam_users, steam_users_table, conn)
    add_or_update(game_reviews, game_reviews_table, conn)


@click.command()
@click.option('--games_file', type=click.Path(exists=True), default=None, help='Path to games file')
@click.option('--review_file', type=click.Path(exists=True), default=None, help='Path to review file')
def main(games_file, review_file):
    # Connect to the database
    engine = create_engine(DATABASE_URI)
    metadata.create_all(engine)
    with engine.connect() as conn:
        if not games_file:
            games_file = 'games_top_100_2weeks.json'
        # Process the files
        process_game_file(games_file, conn)

        if not review_file:
            review_file = 'app_reviews_top_100_2weeks.json'
        process_review_file(review_file, conn)


if __name__ == '__main__':
    main()


