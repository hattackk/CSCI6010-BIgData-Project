import json, os, time

from dotenv import load_dotenv
from sqlalchemy import create_engine
from tqdm import tqdm

from icecream import ic

'''
This might be neat:
from tqdm.contrib.discord import tqdm
for i in tqdm(iterable, token='{token}', channel_id='{channel_id}')
It will add the status bar to discord 
https://tqdm.github.io/docs/contrib.discord/
'''

from database_tables import (
    Base,
    game_review_summary_table, games_table, game_rating_table,
    steam_users_table, game_reviews_table
)
from load_json_to_database import parse_single_game_review, parse_game_json, add_or_update
from steam_api_client import SteamAPIClient


# Sandbox Test
steam_api_client = SteamAPIClient(api_key=os.environ.get('STEAM_API_KEY'))
app_list = steam_api_client.get_app_list()


'''for i,app in enumerate(app_list):
    if app['appid'] == 329070:
        print(f'Found application {app.get("name")} at {i} {app}')
'''

upload_to_db = True
write_json = False

 
def build_set_from_json_file(file_path, key):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
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
                print(f"Error decoding JSON from line: {line}")
                continue

            # Extract the value corresponding to the key
            value = json_obj.get(key)

            # Add the value to the set
            if value is not None:
                key_set.add(value)

    return key_set


# already_processed_apps = build_set_from_json_file('game_reviews.json', 'application_id')
already_processed_apps = None
games = {}
with open('top100in2weeks.json', 'r') as input_games:
    games = json.load(input_games)


apps = list(games)
print(len(apps))
if already_processed_apps is not None:
    temp_apps = []
    for app in apps:
        if app not in already_processed_apps:
            temp_apps.append(app)
    apps = temp_apps

out_of = len(apps)

# set up the db connection
load_dotenv()
usr = os.environ.get('DB_USER')
pwd = os.environ.get("DB_PWD")
db_host = os.environ.get("DB_HOST")
db_port = os.environ.get("DB_PORT")
db_db = os.environ.get("DB_DATABASE")
DATABASE_URI = f'postgresql+psycopg2://{usr}:{pwd}@{db_host}:{db_port}/{db_db}'
engine = create_engine(DATABASE_URI)
Base.metadata.create_all(engine)

for app in tqdm(['570'] + apps):
    time.sleep(5)
    response = steam_api_client.get_reviews_for_app(
        language='english',
        app_id=app, day_range=365,
        num_per_page=100
    )
    games[app]['query_summary'] = response.get('query_summary', "None")
    #  games -> top100in2weeks.json
    #  review -> app_review_top_100.json
    if upload_to_db:
        ic(response.keys())
        ic(response['query_summary'])
        ic(games[app])
        sleep(10)
        # here is looks like the only thing be updated is the query_summary
        # with hits the games_review_summary table only
        with engine.connect() as conn:
            game, game_review_sum, game_rating = parse_game_json({app: games[app]})
            add_or_update(game, games_table, conn)
            add_or_update(game_review_sum, game_review_summary_table, conn)
            add_or_update(game_rating, game_rating_table, conn)

            for review in response.get('reviews', []):
                review['application_id'] = app
                single_review, single_user = parse_single_game_review(review)
                add_or_update([single_user], steam_users_table, conn)
                add_or_update([single_review], game_reviews_table, conn)
    if write_json:
        with open('game_updated.json', 'w') as json_results:
            json.dump(games, json_results)
        with open("game_reviews.json", "a") as json_review_results:
            for review in response.get('reviews', []):
                review['application_id'] = app
                json.dump(review, json_review_results)
                json_review_results.write('\n')

    




