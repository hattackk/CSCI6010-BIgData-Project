from steam_api_client import SteamAPIClient

# Sandbox Test
steam_api_client = SteamAPIClient()
app_list = steam_api_client.get_app_list()


'''for i,app in enumerate(app_list):
    if app['appid'] == 329070:
        print(f'Found application {app.get("name")} at {i} {app}')
'''

import json,os,time

 
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

already_processed_apps=build_set_from_json_file('game_reviews.json','application_id')

games={}
with open ('games.json','r') as input_games:
    games = json.load(input_games)


apps=list(games)
print(len(apps))
if already_processed_apps is not None:
    temp_apps = []
    for app in apps:
        if app not in already_processed_apps:
            temp_apps.append(app)
    apps = temp_apps

out_of = len(apps)
for i,app in enumerate(apps):
    print(f'Working on app {games[app]["name"]}:{app} ({i} of {out_of})                                                                     ',end='\r')
    time.sleep(1)
    response = steam_api_client.get_reviews_for_app(language='english', app_id=app, day_range=365, num_per_page=100)
    games[app]['query_summary']=response.get('query_summary',"None")
    with open('game_updated.json','w') as json_results:
        json.dump(games, json_results)
    with open("game_reviews.json","a") as json_review_results:
        for review in response.get('reviews',[]):
            review['application_id'] = app
            json.dump(review, json_review_results)
            json_review_results.write('\n')

    




