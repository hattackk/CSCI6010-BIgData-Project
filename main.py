from steam_api_client import SteamAPIClient

# Sandbox Test
steam_api_client = SteamAPIClient()
app_list = steam_api_client.get_app_list()


'''for i,app in enumerate(app_list):
    if app['appid'] == 329070:
        print(f'Found application {app.get("name")} at {i} {app}')
'''

import json

import steamspypi

data_request = dict()
data_request['request'] = 'top100in2weeks'
top_100_in_2_week = steamspypi.download(data_request)


apps=list(top_100_in_2_week)

out_of = len(apps)
for i,app in enumerate(apps):
    response = steam_api_client.get_reviews_for_app(app_id=app,day_range=365,num_per_page=100)
    top_100_in_2_week[app]['query_summary']=response.get('query_summary',"None")
    with open("app_reviews_top_100_2weeks.json","a") as json_review_results:
        for review in response.get('reviews',[]):
            review['application_id'] = app
            json.dump(review, json_review_results)
            json_review_results.write('\n')
    print(f'Working on app {i} of {out_of}',end='\r')

print('Saving app descriptions.')
with open('top100in2weeks.json','w') as json_results:
    json.dump(top_100_in_2_week, json_results)
    



