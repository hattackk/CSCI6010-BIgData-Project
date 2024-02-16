import json
import os
import time
import logging
import click
from dotenv import load_dotenv
from sqlalchemy import create_engine, select, update, exc

from database_tables import (
    metadata,
    game_review_summary_table, steam_users_table, game_reviews_table, game_review_download_status_table
)
from load_json_to_database import parse_single_game_review, parse_game_json, add_or_update
from steam_api_client import SteamAPIClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# database and api params
usr = os.environ.get('DB_USER')
pwd = os.environ.get("DB_PWD")
db_host = os.environ.get("DB_HOST")
db_port = os.environ.get("DB_PORT")
db_db = os.environ.get("DB_DATABASE")
DATABASE_URI = f'postgresql+psycopg2://{usr}:{pwd}@{db_host}:{db_port}/{db_db}'
steam_api_client = SteamAPIClient(api_key=os.environ.get('STEAM_API_KEY'))

'''
app_list = steam_api_client.get_app_list()
for i,app in enumerate(app_list):
    if app['appid'] == 329070:
        print(f'Found application {app.get("name")} at {i} {app}')
'''

'''
This might be neat:
from tqdm.contrib.discord import tqdm
for i in tqdm(iterable, token='{token}', channel_id='{channel_id}')
It will add the status bar to discord 
https://tqdm.github.io/docs/contrib.discord/
'''


def build_set_from_json_file(file_path, key):
    # Check if the file exists
    if not os.path.exists(file_path):
        logging.error(f"File '{file_path}' does not exist.")
        return None

    # Initialize an empty set
    key_set = set()

    # Open the file and loop through each line
    with open(file_path, 'r') as file:
        for line in file:
            # Load the JSON object from the line
            try:
                json_obj = json.loads(line.strip())
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from line: {line}")
                continue

            # Extract the value corresponding to the key
            value = json_obj.get(key)

            # Add the value to the set
            if value is not None:
                key_set.add(value)

    return key_set


@click.command()
@click.option('--upload_to_db', is_flag=True, default=False, help='Upload data to the database')
@click.option('--write_json', is_flag=True, default=False, help='Write data to a JSON file')
def main(upload_to_db, write_json):
    engine = create_engine(DATABASE_URI)
    metadata.create_all(engine)
    games = {}
    should_download_reviews = True
    while should_download_reviews:
        stmt = select(game_review_download_status_table.c.game_id).where(
            game_review_download_status_table.c.status == 'not_started'
        ).order_by(game_review_download_status_table.c.game_id)
        with engine.connect() as conn:
            result = conn.execute(stmt).all()
            if not result or len(result) == 0:
                logging.info("No games reviews to download.")
                should_download_reviews = False
                break

            app = result[0][0]
        
        logging.info(f'{len(result)} games remaining, processing {app}')

        update_stmt = update(game_review_download_status_table).where(
            game_review_download_status_table.c.game_id == app
        ).values(
            status='processing'
        )

        # Execute the update statement
        with engine.connect() as conn:
            conn.execute(update_stmt)
            conn.commit()

        time.sleep(0.5)
        response = steam_api_client.get_reviews_for_app(
            language='english',
            app_id=app, day_range=365,
            num_per_page=100
        )
        try:
            games[app]['query_summary'] = response.get('query_summary', "None")
        except KeyError:
            games[app] = dict(query_summary=response.get('query_summary', "None"), appid=app)
        #  games -> top100in2weeks.json
        #  review -> app_review_top_100.json
        if upload_to_db:
            # here is looks like the only thing be updated is the query_summary
            # with hits the games_review_summary table only
            try:
                with engine.connect() as conn:
                    game, game_review_sum, game_rating = parse_game_json({app: games[app]})
                    add_or_update(game_review_sum, game_review_summary_table, conn)

                    for review in response.get('reviews', []):
                        review['application_id'] = app
                        single_review, single_user = parse_single_game_review(review)
                        add_or_update([single_user], steam_users_table, conn)
                        add_or_update([single_review], game_reviews_table, conn)

                    update_stmt = update(game_review_download_status_table).where(
                        game_review_download_status_table.c.game_id == app
                    ).values(
                        status='done'
                    )
                    # Execute the update statement
                    conn.execute(update_stmt)
                    conn.commit()
            except exc.DataError as error:
                logging.error(f'Failed to execute query for app_id {app}\n{error}')
                update_stmt = update(game_review_download_status_table).where(
                game_review_download_status_table.c.game_id == app
                ).values(
                    status='failed'
                )
                # Execute the update statement
                with engine.connect() as conn:
                    conn.execute(update_stmt)
                    conn.commit()

        if write_json:
            data_dir='data'
            with open(os.path.join(data_dir,'game_updated.json'), 'w') as json_results:
                json.dump(games, json_results)
            with open(os.path.join(data_dir,"game_reviews.json"), "a") as json_review_results:
                for review in response.get('reviews', []):
                    review['application_id'] = app
                    json.dump(review, json_review_results)
                    json_review_results.write('\n')


if __name__ == '__main__':
    main()



