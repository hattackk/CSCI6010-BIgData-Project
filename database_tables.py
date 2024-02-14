from sqlalchemy import Column, Integer, Float, String, MetaData, Table
from sqlalchemy import BigInteger, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()
metadata = MetaData()


games_table = Table(
    'games', metadata,
    Column('game_id', Integer, primary_key=True),
    Column('game_name', String, nullable=False, unique=False),
    Column('developer', String),
    Column('publisher', String),
    Column('owners', String),
    Column('price', Float),
    Column('initialprice', Float),
    Column('discount', Float),
    Column('ccu', Integer)
)


game_rating_table = Table(
    'game_rating', metadata,
    Column('game_id', Integer, primary_key=True),
    Column('score_rank', Integer),
    Column('positive', Integer),
    Column('negative', Integer),
    Column('userscore', Integer),
    Column('average_forever', Integer),
    Column('average_2weeks', Integer),
    Column('median_forever', Integer),
    Column('median_2weeks', Integer)
                   )

game_review_summary_table = Table(
    'game_review_summary', metadata,
    Column('game_id', Integer, primary_key=True),
    Column('num_reviews', Integer),
    Column('review_score', Integer),
    Column('review_score_desc', String),
    Column('total_positive', Integer),
    Column('total_negative', Integer),
    Column('total_reviews', Integer)
)


game_reviews_table = Table(
    'game_reviews', metadata,
     Column('recommendationid', BigInteger, primary_key=True),
     Column('author', BigInteger),
     Column('language', String),
     Column('review', String),
     Column('timestamp_created', Integer),
     Column('timestamp_updated', Integer),
     Column('voted_up', Boolean),
     Column('votes_up', Integer),
     Column('votes_funny', Integer),
     Column('weighted_vote_score', Float),
     Column('comment_count', Integer),
     Column('steam_purchase', Boolean),
     Column('received_for_free', Boolean),
     Column('written_during_early_access', Boolean),
     Column('hidden_in_steam_china', Boolean),
     Column('steam_china_location', String),
     Column('application_id', Integer, ForeignKey('games.game_id'))
)


steam_users_table = Table(
    'steam_users', metadata,
    Column('steamid', BigInteger, primary_key=True),
    Column('num_games_owned', Integer),
    Column('num_reviews', Integer),
    Column('playtime_forever', Integer),
    Column('playtime_last_two_weeks', Integer),
    Column('playtime_at_review', Integer),
    Column('last_played', Integer, ForeignKey('games.game_id'))
)