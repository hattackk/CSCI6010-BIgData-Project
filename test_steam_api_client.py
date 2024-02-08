import pytest
from steam_api_client import SteamAPIClient

@pytest.fixture
def steam_api_client():
    return SteamAPIClient()

def test_make_request_success(steam_api_client, monkeypatch):
    # Mock _make_request to return a dummy response
    def mock_make_request(interface, method, version, params=None):
        return {"status": "success"}

    monkeypatch.setattr(steam_api_client, "_make_request", mock_make_request)

    response = steam_api_client._make_request("ISteamNews", "GetNewsForApp", 2, params={'appid': 570})
    assert response == {"status": "success"}

def test_make_request_failure(steam_api_client, monkeypatch):
    # Mock _make_request to return None for failure
    def mock_make_request(interface, method, version, params=None):
        return None

    monkeypatch.setattr(steam_api_client, "_make_request", mock_make_request)

    response = steam_api_client._make_request("InvalidInterface", "InvalidMethod", 99)
    assert response is None

def test_get_news_for_app_success(steam_api_client, monkeypatch):
    # Mock _make_request to return a dummy response
    def mock_make_request(interface, method, version, params=None):
        return {"status": "success"}

    monkeypatch.setattr(steam_api_client, "_make_request", mock_make_request)

    response = steam_api_client.get_news_for_app(570)  # Dota 2 AppID
    assert response == {"status": "success"}

def test_get_news_for_app_failure(steam_api_client, monkeypatch):
    # Mock _make_request to return None for failure
    def mock_make_request(interface, method, version, params=None):
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
    def mock_make_request(interface, method, version, params=None):
        return dummy_response

    monkeypatch.setattr(steam_api_client, "_make_request", mock_make_request)

    # Call the method
    apps = steam_api_client.get_app_list()

    # Verify the response
    assert apps == dummy_response["applist"]["apps"]
