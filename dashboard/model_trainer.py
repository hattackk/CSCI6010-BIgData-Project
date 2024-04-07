import os
import psycopg2

from model import RecommenderModel
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from model import RecommenderModel
from numpy.random import randint
from tabulate import tabulate

load_dotenv()
pd.set_option('display.max_columns', None)

def load_dataframe_from_pickle(file_path):
    try:
        df = pd.read_pickle(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

class ModelTrainer():

    def __init__(self, name='test'):
        ### Review Similarities
        table_names = ["games", "game_reviews", "game_rating", "game_review_summary", "steam_users", "app_type"]

        dataframes = {}
        for table_name in table_names:
            file_path = f"cache/{table_name}.pkl"
            df = load_dataframe_from_pickle(file_path)
            if df is not None:
                dataframes[table_name] = df

        self.users_df=dataframes['steam_users']
        game_reviews_df=dataframes['game_reviews']
        game_rating=dataframes['game_rating']
        game_review_summary=dataframes['game_review_summary']
        print(f"total reviews == {len(game_reviews_df)}")
        game_df=dataframes['games']
        types_df=dataframes['app_type']


        game_reviews_df = game_reviews_df.astype({'voted_up': int})
        game_reviews_df.loc[game_reviews_df['voted_up'] == 0, 'voted_up'] = -1


        game_df = game_df.loc[:, ~game_df.columns.duplicated()].copy()

        # make sure both dfs have same games
        game_reviews_df = game_reviews_df.query('application_id in @game_df.game_id.unique()')
        game_df = game_df.query('game_id in @game_reviews_df.application_id.unique()')
        game_reviews_df = game_reviews_df.query('application_id in @game_df.game_id.unique()')

        # add types cols to games
        game_df = pd.merge(game_df, game_rating, on='game_id', how='left')
        game_df = pd.merge(game_df, game_review_summary, on='game_id', how='left')
        game_df = pd.merge(game_df, types_df, left_on='game_id', right_on='app_id', how='left')
        game_df['genres']=game_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
        game_df['categories']=game_df['categories'].apply(lambda x: x if isinstance(x, list) else [])

        # clean datas
        # df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill NaN with a default value, e.g., 0
        # df.fillna(0, inplace=True)
        # df['app_id'] = df['app_id'].astype(int)
        # df['metacritic'] = df['metacritic'].astype(int)
        
        game_df = game_df.loc[:, ~game_df.columns.duplicated()].copy()

        self.game_df=game_df
        self.game_reviews_df = game_reviews_df
        self.model_name=f'{name}.pkl.xz'
        self.all_numerical_cols=['num_reviews', 'review_score', 'total_positive', 'total_negative', 'price']
        self.all_categorical_cols=['review_score_desc', 'developer', 'publisher', 'owners']
        self.all_multi_label_cols=['genres', 'categories']
        self.numerical_cols=self.all_numerical_cols
        self.categorical_cols=self.all_categorical_cols
        self.multi_label_cols=self.all_multi_label_cols
        self.test_users = []

    def set_numerical_cols(self, selection):
        self.numerical_cols = selection

    def set_categorical_cols(self, selection):
        self.categorical_cols = selection

    def set_multi_cols(self, selection):
        self.multi_label_cols = selection

    def get_possible_numerical_cols(self):
        return self.all_numerical_cols

    def get_possible_categorical_cols(self):
        return self.all_categorical_cols

    def get_possible_multi_cols(self):
        return self.all_multi_label_cols
    
    def set_model_name(self, name):
        self.model_name = name

    def execute_train(self, test=None):
        model = RecommenderModel()

        # only pass cols in selected columns (and ids)
        pd.set_option('display.max_columns', None)
        all_selected = self.numerical_cols+self.categorical_cols+self.multi_label_cols
        all_defaults = self.all_numerical_cols + self.all_categorical_cols + self.all_multi_label_cols
        not_selected = [col for col in all_defaults if col not in all_selected]
        train_game_df = self.game_df.copy()
        train_game_df=train_game_df.drop(columns=not_selected)

        train_reviews_df = self.game_reviews_df

        if test is not None and test > 0:
            filtered_users_df = self.users_df[self.users_df['num_reviews'] > 1]
            merged_df = pd.merge(train_reviews_df, filtered_users_df, on='steamid')
            voted_up_df = merged_df[merged_df['voted_up'] == 1]
            test_df = voted_up_df.sample(n=test, random_state=42)  # Use a random state for reproducibility
            test_indices = test_df.index
            train_reviews_df = train_reviews_df.drop(test_indices)
            self.test_users=test_df
        model.train_game_similarities(train_game_df, numerical_cols=self.numerical_cols, categorical_cols=self.categorical_cols, multi_label_cols=self.multi_label_cols)
        model.train_user_similaries(train_reviews_df)
        model.save(self.model_name)
        self.model = RecommenderModel.load(self.model_name)

    def test_row(self, row):
        recommendation=pd.DataFrame([{'game_id': 'NaN'}])
        num_recs=1
        while len(recommendation[recommendation['game_id'] == row['application_id']]) < 1:
            recommendation = self.model.predict(row['steamid'], num_recommendations=num_recs)
            num_recs+=5
        return recommendation
    

    def test_eval(self):
        results=[]
        # Compute the result and check for each user
        for index, row in self.test_users.iterrows():
            rec = self.test_row(row)
            target = row['application_id']
            find_target_df = self.game_df[self.game_df['game_id']==target]
            target_game=''
            if not find_target_df.empty:
                target_game = find_target_df.iloc[0]['game_name']
            else:
                print('ERROR: Test target application ID does not match any games.')
                continue # no matching game for this review??
            got_expected = len(rec[rec['game_id'] == target]) > 0

            # Append to new_data
            results.append({'test_game': target_game,'application_id': target, 'got_expected': got_expected,
                            'recommendations': rec['game_name'], 'result': len(rec)})

        return pd.DataFrame(results)

if __name__ == '__main__':
    trainer = ModelTrainer()
    # trainer.set_multi_cols([])
    trainer.execute_train(test=30)
    trainer.test_eval()


