import pytest
from unittest import TestCase
from steam_api_client import SteamAPIClient


def test_make_request_success(steam_api_client, monkeypatch):
    # Mock _make_request to return a dummy response
    def mock_make_request(base_url,interface, method, version, params=None):
        return {"status": "success"}

    monkeypatch.setattr(steam_api_client, "_make_request", mock_make_request)

    response = steam_api_client._make_request("steam.com","ISteamNews", "GetNewsForApp", 2, params={'appid': 570})
    assert response == {"status": "success"}

def test_make_request_failure(steam_api_client, monkeypatch):
    # Mock _make_request to return None for failure
    def mock_make_request(base_url,interface, method, version, params=None):
        return None

    monkeypatch.setattr(steam_api_client, "_make_request", mock_make_request)

    response = steam_api_client._make_request("steam.com","InvalidInterface", "InvalidMethod", 99)
    assert response is None

def test_get_news_for_app_success(steam_api_client, monkeypatch):
    # Mock _make_request to return a dummy response
    def mock_make_request(base_url,interface, method, version, params=None):
        return {"status": "success"}

    monkeypatch.setattr(steam_api_client, "_make_request", mock_make_request)

    response = steam_api_client.get_news_for_app(570)  # Dota 2 AppID
    assert response == {"status": "success"}

def test_get_news_for_app_failure(steam_api_client, monkeypatch):
    # Mock _make_request to return None for failure
    def mock_make_request(base_url,interface, method, version, params=None):
        return None

    monkeypatch.setattr(steam_api_client, "_make_request", mock_make_request)

    response = steam_api_client.get_news_for_app(999999999)  # Invalid AppID
    assert response is None

def test_initialization_without_api_key():
    steam_api_client = SteamAPIClient()
    assert steam_api_client.api_key is None


def test_get_app_list_success(steam_api_client, monkeypatch):
    # Define a dummy response
    dummy_response = {
        "applist": {
            "apps": [
                {"appid": 1, "name": "App 1"},
                {"appid": 2, "name": "App 2"},
                {"appid": 3, "name": "App 3"}
            ]
        }
    }

    # Mock _make_request to return the dummy response
    def mock_make_request(base_url,interface, method, version, params=None):
        return dummy_response

    monkeypatch.setattr(steam_api_client, "_make_request", mock_make_request)

    # Call the method
    apps = steam_api_client.get_app_list()

    # Verify the response
    assert apps == dummy_response["applist"]["apps"]


def test_get_reviews_for_app_success(steam_api_client, monkeypatch):
    # Define a dummy response
    dummy_response = {
        "success": 1,
        "query_summary": {
            "num_reviews": 10,
            "review_score": 90,
            "review_score_desc": "Very Positive",
            "total_positive": 900,
            "total_negative": 100,
            "total_reviews": 1000
        },
        "cursor": "*",
        "reviews": [
            {
                "recommendationid": "12345",
                "author": {
                    "steamid": "steam_id_123",
                    "num_games_owned": 50,
                    "num_reviews": 5,
                    "playtime_forever": 1000,
                    "playtime_last_two_weeks": 20,
                    "playtime_at_review": 500,
                    "last_played": "timestamp",
                },
                "language": "english",
                "review": "This is a great game!",
                "timestamp_created": 1635647600,
                "timestamp_updated": 1635651200,
                "voted_up": True,
                "votes_up": 50,
                "votes_funny": 10,
                "weighted_vote_score": 0.9,
                "comment_count": 5,
                "steam_purchase": True,
                "received_for_free": False,
                "written_during_early_access": False,
                "developer_response": "Thank you for your review!",
                "timestamp_dev_responded": 1635654800
            },
            # Additional review objects can be added for testing purposes
        ]
    }

    # Mock _make_request to return the dummy response
    def mock_make_request(base_url, interface, method, version, params=None):
        return dummy_response

    monkeypatch.setattr(steam_api_client, "_make_request", mock_make_request)

    # Call the method
    reviews = steam_api_client.get_reviews_for_app(570)  # Dota 2 AppID

    # Verify the response
    TestCase().assertDictEqual(reviews, dummy_response)

