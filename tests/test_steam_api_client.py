import pytest
from unittest import TestCase
from steam_api_client import SteamAPIClient


def test_make_request_success(steam_api_client):
    response = steam_api_client._make_request("https://store.steampowered.com/", interface="appreviews")
    assert response == {"success": 2}


def test_make_request_failure(steam_api_client):
    # Mock _make_request to return None for failure

    response = steam_api_client._make_request("https://steam.com", "InvalidInterface", "InvalidMethod", 99)
    assert response is None


def test_get_news_for_app_success(steam_api_client):
    response = steam_api_client.get_news_for_app(570)  # Dota 2 AppID
    assert 'appnews' in response.keys()
    assert response['appnews']['count'] > 0


def test_get_news_for_app_failure(steam_api_client):
    response = steam_api_client.get_news_for_app(-1)  # Invalid AppID
    assert response is None


def test_initialization_without_api_key():
    steam_api_client = SteamAPIClient()
    assert steam_api_client.api_key is None


def test_get_app_list_success(steam_api_client):
    apps = steam_api_client.get_app_list()

    # Verify the response
    assert len(apps) > 0
    for app in apps:
        assert len(app.keys()) == 2
        assert 'appid' in app.keys()
        assert 'name' in app.keys()


def test_get_reviews_for_app_success(steam_api_client):
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
                "hidden_in_steam_china": True,
                "steam_china_location": ""

            },
            # Additional review objects can be added for testing purposes
        ]
    }

    # Call the method
    reviews = steam_api_client.get_reviews_for_app(570, num_per_page=1)  # Dota 2 AppID

    # verify the keys
    assert set(reviews.keys()) == set(dummy_response.keys())
    assert set(reviews['reviews'][0].keys()) == set(dummy_response['reviews'][0].keys())
    assert set(reviews['reviews'][0]['author'].keys()) == set(dummy_response['reviews'][0]['author'].keys())
    assert set(reviews['query_summary'].keys()) == set(dummy_response['query_summary'].keys())


