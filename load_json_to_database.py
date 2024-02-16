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
    for record in record_list:
        primary_keys = [key.name for key in inspect(table).primary_key]
        stmt = insert(table).values(record)
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


def process_game_file(games_file, conn):
    # load the games_file
    with open(games_file, 'r') as f:
        games_file = json.loads(f.read())

    games, games_review_summary, game_rating = parse_game_json(games_file)
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


