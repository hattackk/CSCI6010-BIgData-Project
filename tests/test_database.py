from sqlalchemy import inspect
from database_tables import (
    games_table,
    game_rating_table,
    game_review_summary_table,
    game_reviews_table,
    steam_users_table)


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
