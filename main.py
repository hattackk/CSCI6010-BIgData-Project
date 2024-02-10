from steam_api_client import SteamAPIClient

# Sandbox Test
steam_api_client = SteamAPIClient()
app_list = steam_api_client.get_app_list()


for i,app in enumerate(app_list):
    if app['appid'] == 329070:
        print(f'Found application {app.get("name")} at {i} {app}')

import json
response = steam_api_client.get_reviews_for_app(app_id=329070, num_per_page=1)
print(response.get('reviews'))

    

