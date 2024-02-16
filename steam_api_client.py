import requests
import warnings
import logging

from urllib.parse import quote
import requests.exceptions as req_exc


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        self.reviews_base_url = "https://store.steampowered.com/"
        self.api_key = api_key

    def _make_request(self, base_url, interface=None, method=None, version=None, params=None):
        """
        Makes a request to the Steam Web API.

        Parameters:
            base_url (str): The base URL of the API.
            interface (str, optional): The interface of the API.
            method (str, optional): The method of the API.
            version (int, optional): The version of the API.
            params (dict, optional): Additional parameters to include in the request.

        Returns:
            dict or None: The JSON response from the API, or None if the request fails.
        """
        url = f"{base_url}/"
        if interface:
            url += f"{interface}/"
        if method:
            url += f"{method}/"
        if version:
            url += f"v{version}/"
        if self.api_key:
            params = params or {}
            params['key'] = self.api_key

        try:
            response = requests.get(url, params=params)
        except req_exc.ConnectionError as e:
            logging.error(f"Failed to make request: {e}")
            return None

        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Failed to make request. Status code: {response.status_code}")
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
        return self._make_request(self.base_url, "ISteamNews", "GetNewsForApp", 2, params=params)

    def get_app_list(self):
        """
        Retrieves the complete list of public apps from the Steam Web API.

        Returns:
            list: A list of dictionaries, each representing an application.
                  Each dictionary contains the following keys:
                    - appid (int): The App ID of the application.
                    - name (str): The name of the application.
        """
        response = self._make_request(self.base_url, "ISteamApps", "GetAppList", 2)
        if response:
            return response.get('applist', {}).get('apps', [])
        else:
            return []

    def get_reviews_for_app(self, app_id, filter="all", language="all", day_range=365, cursor="*",
                            review_type="all", purchase_type="steam", num_per_page=20, filter_offtopic_activity=None):
        """
        Retrieves reviews for a specific app from the Steam Web API.

        Parameters:
            app_id (int): The AppID of the game or application.
            filter (str): The filter for reviews (default is "all").
            language (str): The language of reviews (default is "all").
            day_range (str): The range from now to n days ago to look for helpful reviews.
            cursor (str): The cursor value to retrieve the next batch of reviews (default is "*").
            review_type (str): The type of reviews (default is "all").
            purchase_type (str): The purchase type of reviews (default is "steam").
            num_per_page (int): The number of reviews per page (default is 20).
            filter_offtopic_activity (bool): Whether to include off-topic reviews (default is None).

        Returns:
            dict or None: The JSON response containing reviews, or None if the request fails.
        """
        if day_range > 365 or day_range < 1:
            warnings.warn(f"Invalid value {day_range} for parameter day_range. Value must be between 1 and 365.")
            day_range = 365
        if num_per_page > 100 or num_per_page < 1:
            warnings.warn(f"Invalid value {num_per_page} for parameter num_per_page. Value must be between 1 and 100.")
            num_per_page = 20
        if cursor != '*':
            cursor = quote(cursor)
        params = {
            "json": "1",
            'filter': filter,
            'language': language,
            'day_range': day_range,
            'cursor': cursor,
            'review_type': review_type,
            'purchase_type': purchase_type,
            'num_per_page': num_per_page,
            'filter_offtopic_activity': filter_offtopic_activity
        }
        return self._make_request(
            self.reviews_base_url, interface="appreviews",
            method=str(app_id), version=None, params=params
        )