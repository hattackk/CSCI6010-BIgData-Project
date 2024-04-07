import os
import psycopg2

from model import RecommenderModel
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':


    ### Review Similarities

    # Database connection parameters
    usr = os.environ.get('DB_USER')
    if usr is None:
        raise Exception("Environment variable 'DB_USER' not found. Please set it and try again.")

    pwd = os.environ.get("DB_PWD")
    if pwd is None:
        raise Exception("Environment variable 'DB_PWD' not found. Please set it and try again.")

    db_host = os.environ.get("DB_HOST")
    if db_host is None:
        raise Exception("Environment variable 'DB_HOST' not found. Please set it and try again.")

    db_port = os.environ.get("DB_PORT")
    if db_port is None:
        raise Exception("Environment variable 'DB_PORT' not found. Please set it and try again.")

    db_db = os.environ.get("DB_DATABASE")
    if db_db is None:
        raise Exception("Environment variable 'DB_DATABASE' not found. Please set it and try again.")

    conn = psycopg2.connect(
        dbname=db_db,
        user=usr,
        password=pwd,
        host=db_host,
        port=db_port
    )

    cur = conn.cursor()

    cur.execute("SELECT steamid, application_id, voted_up FROM game_reviews")
    rows = cur.fetchall()
    # Convert to pandas DataFrame
    df_reviews = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
    df_reviews = df_reviews.astype({'voted_up': int})
    df_reviews.loc[df_reviews['voted_up'] == 0, 'voted_up'] = -1

    ### Game Similarities
    nc = ['num_reviews', 'review_score', 'total_positive', 'total_negative', 'price']
    cc = ['review_score_desc', 'developer', 'publisher', 'owners']
    columns = nc + cc
    query = f"""
    SELECT *
    FROM games 
    join game_rating 
    on games.game_id = game_rating.game_id 
    join game_review_summary 
    on game_review_summary.game_id = games.game_id 
    WHERE games.game_id IN (SELECT application_id FROM game_reviews)
    """
    cur.execute(query)
    rows = cur.fetchall()
    df_games = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
    df_games = df_games.loc[:, ~df_games.columns.duplicated()].copy()

    model = RecommenderModel()
    model.train_game_similarities(df_games, multi_label_cols=[])
    model.train_user_similaries(df_reviews)
    model.save('recommender_model.pkl.xz')

