import requests

class SteamAPIClient:
    """
    A client for interacting with the Steam Web API.
    """

    def __init__(self, api_key=None):
        """
        Initializes the SteamAPIClient.

        Parameters:
            api_key (str, optional): The API key for accessing the Steam Web API.
        """
        self.base_url = "https://api.steampowered.com"
        self.api_key = api_key

    def _make_request(self, interface, method, version, params=None):
        """
        Makes a request to the Steam Web API.

        Parameters:
            interface (str): The interface of the API.
            method (str): The method of the API.
            version (int): The version of the API.
            params (dict, optional): Additional parameters to include in the request.

        Returns:
            dict or None: The JSON response from the API, or None if the request fails.
        """
        url = f"{self.base_url}/{interface}/{method}/v{version}/"
        if self.api_key:
            params = params or {}
            params['key'] = self.api_key
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to make request. Status code: {response.status_code}")
            return None

    def get_news_for_app(self, app_id, count=3):
        """
        Retrieves news articles for a specific app from the Steam Web API.

        Parameters:
            app_id (int): The AppID of the game or application.
            count (int, optional): The number of news articles to retrieve (default is 3).

        Returns:
            dict or None: The JSON response containing news articles, or None if the request fails.
        """
        params = {
            'appid': app_id,
            'count': count
        }
        return self._make_request("ISteamNews", "GetNewsForApp", 2, params=params)
    
    def get_app_list(self):
        """
        Retrieves the complete list of public apps from the Steam Web API.

        Returns:
            list: A list of dictionaries, each representing an application.
                  Each dictionary contains the following keys:
                    - appid (int): The App ID of the application.
                    - name (str): The name of the application.
        """
        response = self._make_request("ISteamApps", "GetAppList", 2)
        if response:
            return response.get('applist', {}).get('apps', [])
        else:
            return []



