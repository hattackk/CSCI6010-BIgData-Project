from steam_api_client import SteamAPIClient

# Sandbox Test
steam_api_client = SteamAPIClient()
app_list = steam_api_client.get_app_list()


for i,app in enumerate(app_list):
    if app['appid'] == 1091500:
        print(f'Found Cyberpunk 2077 at {i} {app}')

import json
print(json.dumps(steam_api_client.get_news_for_app(1091500),indent=4))

    

