# CSCI6010-BigData-Project

# Steam API Client

This Python package provides a client for interacting with the Steam Web API, allowing users to retrieve various data from the Steam platform such as game information, user stats, reviews, and more.

## Installation

There's no installation required as this is a Python package used for interacting with the Steam Web API. You can simply clone the repository and import the `SteamAPIClient` class into your project.

```bash
git clone https://github.com/hattackk/CSCI6010-BIgData-Project.git
```

## Usage

### Initialization

To use the Steam API Client, you need to initialize an instance of the `SteamAPIClient` class. You can optionally provide your Steam API key during initialization:

```python
from steam_api_client import SteamAPIClient

# Initialize the client without an API key
steam_client = SteamAPIClient()

# Initialize the client with your API key
steam_client_with_key = SteamAPIClient(api_key='YOUR_API_KEY')
```

### Retrieving App List

You can retrieve the complete list of public apps from the Steam Web API using the `get_app_list` method:

```python
apps = steam_client.get_app_list()
print(apps)
```
This will return a list of dictionaries, each representing an application with keys 'appid' and 'name'.
### Load jsons into postgres instance

Once you have a json with the review or game data, you can load it into the postgress instance:
```
python load_json_to_database.py  --games_file path_to_games_file.json
python load_json_to_database.py --review_file path_to_review_file.json
python load_json_to_database.py --games_file path_to_games_file.json --review_file path_to_review_file.json
```
The loading tool uses the `dot_env` package to parse a `.env` file which should be included in the root directory 
but ***NOT*** added to the github repo. here is an example:
```
DB_USER="{USERNAME}"
DB_PWD="{PASSWORD}"
DB_HOST="{HOST_NAME}"
DB_PORT="{PORT}"2`
DB_DATABASE="{DB NAME}"
```

### Retrieving News for an App

You can retrieve news articles for a specific app using the `get_news_for_app` method:

```python
news = steam_client.get_news_for_app(app_id)
print(news)
```

### Handling Errors

If an error occurs during the API request, the methods will return `None`. Make sure to handle this case in your code.


# Dashboards
### Training and Recommendations are handled through the dashboard services in the /dashboard directory

## Training 

<img width="1292" alt="image" src="https://github.com/hattackk/CSCI6010-BIgData-Project/assets/36685328/b9ce7223-18d8-451e-aabd-c40cfefa226b">

## Recommendations

<img width="1292" alt="image" src="https://github.com/hattackk/CSCI6010-BIgData-Project/assets/36685328/ef2c4a1c-c1ed-448b-b5be-6ba0eea13813">


