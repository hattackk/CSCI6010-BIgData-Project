from sqlalchemy import inspect


def test_connection(connection):
    assert connection


def test_tables_exist(connection):
    inspector = inspect(connection.engine)
    tables = inspector.get_table_names()
    assert 'games' in tables
    assert 'game_rating' in tables
    assert 'game_review_summary' in tables
    assert 'game_reviews' in tables
    assert 'steam_users' in tables
