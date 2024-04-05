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

load_dotenv()
logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# database and api params
usr = os.environ.get('DB_USER')
pwd = os.environ.get("DB_PWD")
db_host = os.environ.get("DB_HOST")
db_port = os.environ.get("DB_PORT")
db_db = os.environ.get("DB_DATABASE")
DATABASE_URI = f'postgresql+psycopg2://{usr}:{pwd}@{db_host}:{db_port}/{db_db}'
steam_api_client = SteamAPIClient(api_key=os.environ.get('STEAM_API_KEY'))


def select_and_update_game_status_to_processing(conn, game_review_download_status_table):
    """
    Selects a game ID and updates its status to 'processing' atomically.

    Args:
        conn: The SQLAlchemy connection object.
        game_review_download_status_table: The SQLAlchemy Table object representing the game review download status table.

    Returns:
        int: The game ID that was selected and marked as processing.
    """
    # Begin a new transaction if not already in one
    if not conn.in_transaction():
        conn.begin()

    try:
        # Select a game ID that has status 'not_started' and mark it as 'processing'
        subquery = (
            select(game_review_download_status_table.c.game_id)
            .where(game_review_download_status_table.c.status == 'not_started')
            .limit(1)
            .with_for_update()  # Lock the selected row for update
            .scalar_subquery()  # Convert the subquery to a scalar subquery
        )

        stmt = (
            update(game_review_download_status_table)
            .values(status='processing')
            .where(game_review_download_status_table.c.game_id == subquery)
            .returning(game_review_download_status_table.c.game_id)
        )

        result = conn.execute(stmt)
        game_id = result.scalar()

        # Commit the transaction
        conn.commit()

        return game_id

    except Exception as e:
        # Rollback the transaction if an error occurs
        conn.rollback()
        raise e

def build_set_from_json_file(file_path, key):
    # Check if the file exists
    if not os.path.exists(file_path):
        logger.error(f"File '{file_path}' does not exist.")
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
                logger.error(f"Error decoding JSON from line: {line}")
                continue

            # Extract the value corresponding to the key
            value = json_obj.get(key)

            # Add the value to the set
            if value is not None:
                key_set.add(value)

    return key_set

def configure_logging(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level,int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.getLogger(__name__).setLevel(log_level)
                        
@click.command()
@click.option('--download_details', is_flag=True, default=False, help='Should download details.')
@click.option('--download_reviews', is_flag=True, default=False, help='Should download reviews.')
@click.option('--upload_to_db', is_flag=True, default=False, help='Upload data to the database')
@click.option('--write_json', is_flag=True, default=False, help='Write data to a JSON file')
@click.option('--log_level', is_flag=False, default="INFO", help='Sets log level')
@click.option('--back_off_timer', is_flag=False, default=60, help='Seconds to sleep if response fails.')

def main(download_details,download_reviews,upload_to_db, write_json, log_level, back_off_timer ):
    if download_details:
        download_details(upload_to_db, write_json, log_level, back_off_timer)
    elif download_reviews:
        download_reviews(upload_to_db, write_json, log_level, back_off_timer)
    else:
        logger.error('No operation given')

def download_details(upload_to_db, write_json, log_level, back_off_timer):
    configure_logging(log_level)
    logger.debug(DATABASE_URI)
    engine = create_engine(DATABASE_URI)
    if not engine:
        logger.error(f"Failed to create engine with URI {DATABASE_URI}")
        return

    metadata.create_all(engine)
    should_download_details = True
    while should_download_details:
        # game_review_download_status_table contains list of all games (in our dataset)
        # select all the games that have not been started
        stmt = select(game_review_download_status_table.c.game_id) \
                .where(game_review_download_status_table.c.status == 'not_started') \
                .order_by(game_review_download_status_table.c.game_id)
        with engine.connect() as conn:
            result = conn.execute(stmt).all()
            if not result or len(result) == 0:
                logger.warning("No games details to download.")
                should_download_details = False
                break
            conn.commit()
            app = select_and_update_game_status_to_processing(conn,game_review_download_status_table)
        
            logger.info(f'{len(result)} games details remaining, processing {app}')

            time.sleep(0.5)
            response = steam_api_client.get_app_details(app)
            # Keep retrying until we get a response.
            while not response:
                logger.warning(f'No response received for {app}. Backing off for {back_off_timer} seconds.')
                time.sleep(back_off_timer)
                response = steam_api_client.get_app_details(app)
            logger.debug(response)
            try:
                pass
            except Exception as e:
                logger.error(f"Error processing response for app {app}: {e}")

            logger.debug(f"Back off timer : {back_off_timer}")
            if upload_to_db:
                with open('details.txt', 'a') as file:
                    file.write(str((app,response))+'\n')
                    
            else:
                # If no operation given, set the application status back to not_started
                logger.warning("No option given, details will not be saved.")
                logger.debug(f"Resetting status of {app}")
                update_stmt = update(game_review_download_status_table).where(
                    game_review_download_status_table.c.game_id == app
                ).values(
                    status='not_started'
                )
                # Execute the update statement
                conn.execute(update_stmt)
                conn.commit()


def download_reviews(upload_to_db, write_json, log_level, back_off_timer):
    configure_logging(log_level)
    logger.debug(DATABASE_URI)
    engine = create_engine(DATABASE_URI)
    if not engine:
        logger.error(f"Failed to create engine with URI {DATABASE_URI}")
        return

    metadata.create_all(engine)
    games = {}
    should_download_reviews = True
    while should_download_reviews:
        # game_review_download_status_table contains list of all games (in our dataset)
        # select all the games that have not been started
        stmt = select(game_review_download_status_table.c.game_id) \
                .where(game_review_download_status_table.c.status == 'not_started') \
                .order_by(game_review_download_status_table.c.game_id)
        with engine.connect() as conn:
            result = conn.execute(stmt).all()
            if not result or len(result) == 0:
                logger.warning("No games reviews to download.")
                should_download_reviews = False
                break
            conn.commit()
            app = select_and_update_game_status_to_processing(conn,game_review_download_status_table)
        
            logger.info(f'{len(result)} games remaining, processing {app}')

            time.sleep(0.5)
            response = steam_api_client.get_reviews_for_app(
                language='english',
                app_id=app, day_range=365,
                num_per_page=100
            )
            # Keep retrying until we get a response.
            while not response:
                logger.warning(f'No response recieved for {app}. Backing of for {back_off_timer} seconds.')
                time.sleep(back_off_timer)
                response = steam_api_client.get_reviews_for_app(
                language='english',
                app_id=app, day_range=365,
                num_per_page=100,
                filter="all",
                cursor=response.get('cursor','*')
            )
            logger.debug(response)

            try:
                games[app]['query_summary'] = response.get('query_summary', "None")
            except KeyError:
                games[app] = dict(query_summary=response.get('query_summary', "None"), appid=app)
            #  games -> games_top_100_2weeks.json
            #  review -> app_reviews_top_100_2weeks.json
                
            logger.debug(f"Back off timer : {back_off_timer}")
            if upload_to_db:
                # here is looks like the only thing be updated is the query_summary
                # with hits the games_review_summary table only
                try:
                    with engine.connect() as conn:
                        game, game_review_sum, game_rating = parse_game_json({app: games[app]})
                        add_or_update(game_review_sum, game_review_summary_table, conn)
                        batch_count = 0
                        seen_cursors = {}
                        while response and response.get('cursor','*') not in seen_cursors:
                            batch_count+=1
                            logger.info(f"Processing App {app} batch {batch_count}")
                            seen_cursors[response.get('cursor','*')] = True
                            for review in response.get('reviews', []):
                                review['application_id'] = app
                                single_review, single_user = parse_single_game_review(review)
                                add_or_update([single_user], steam_users_table, conn)
                                add_or_update([single_review], game_reviews_table, conn)
                            
                            time.sleep(0.5)
                            response = steam_api_client.get_reviews_for_app(
                            language='english',
                            app_id=app, day_range=365,
                            num_per_page=100,
                            filter="all",
                            cursor=response.get('cursor','*')
                            )
                            # Keep Retrying on no response.
                            while not response:
                                logger.warning(f'No response recieved for {app}. Backing of for {back_off_timer} seconds.')
                                time.sleep(back_off_timer)
                                response = steam_api_client.get_reviews_for_app(
                                language='english',
                                app_id=app, day_range=365,
                                num_per_page=100,
                                filter="all",
                                cursor=response.get('cursor','*')
                                )
                        update_stmt = update(game_review_download_status_table).where(
                            game_review_download_status_table.c.game_id == app
                        ).values(
                            status='done'
                        )
                        # Execute the update statement
                        conn.execute(update_stmt)
                        conn.commit()
                except exc.DataError as error:
                    logger.error(f'Failed to execute query for app_id {app}\n{error}')
                    update_stmt = update(game_review_download_status_table).where(
                    game_review_download_status_table.c.game_id == app
                    ).values(
                        status='failed'
                    )
                    # Execute the update statement
                    with engine.connect() as conn:
                        conn.execute(update_stmt)
                        conn.commit()
            else: # no operation given so we must be testing set the application back to not_started. We don't want to have an invalid state in the database because a result will never be uploaded back.
                logger.warning("No option given reviews will not be saved.")
                logger.debug(f"Resetting status of {app}")
                update_stmt = update(game_review_download_status_table).where(
                    game_review_download_status_table.c.game_id == app
                ).values(
                    status='not_started'
                )
                # Execute the update statement
                conn.execute(update_stmt)
                conn.commit()



if __name__ == '__main__':
    main()