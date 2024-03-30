from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import os
import psycopg2
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

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

cur.execute("SELECT steamid, application_id, sentiment_score FROM game_reviews")
rows = cur.fetchall()
# Convert to pandas DataFrame
df_reviews = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])


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
"""
cur.execute(query)
rows = cur.fetchall()
df_games = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
df_games = df_games.loc[:, ~df_games.columns.duplicated()].copy()


class RecommenderModel:
    def __init__(self, player_similarity_weight=0.5):
        self.game_nn_model = None
        self.player_matrix = None
        self.user_index_mapping = None
        self.game_index_mapping = None
        self.user_game_matrix_csr = None
        self.player_similarity_weight = player_similarity_weight
        self.game_similarity_weight = 1 - player_similarity_weight
        self.game_name_df = None

    def train_user_similaries(self, df, n_components=50):
        # TODO: This is broken, fix it.
        """
        trains the TruncatedSVD matrix for user similarities based on sentiment score
        :param df: dataframe with steamid, application_id, sentiment_score
        :return: None
        """
        df = df.copy()
        # use a sparse matrix representation of the matrix
        df['steamid'] = df['steamid'].astype("category")
        df['game_id'] = df['application_id'].astype("category")

        row_ind = df['steamid'].cat.codes
        col_ind = df['game_id'].cat.codes

        user_game_matrix_sparse = coo_matrix(
            (df['sentiment_score'], (row_ind, col_ind)),
            shape=(df['steamid'].nunique(), df['game_id'].nunique())
        )
        user_game_matrix_csr = user_game_matrix_sparse.tocsr()

        svd = TruncatedSVD(n_components=n_components, random_state=42)

        player_matrix = svd.fit_transform(user_game_matrix_csr)
        user_index_mapping = {steamid: index for index, steamid in enumerate(df['steamid'].cat.categories)}
        game_index_mapping = {index: game_id for index, game_id in enumerate(df['game_id'].cat.categories)}
        self.player_matrix = player_matrix
        self.user_game_matrix_csr = user_game_matrix_csr
        self.user_index_mapping = user_index_mapping
        self.game_index_mapping = game_index_mapping

    def train_game_similarities(
            self,
            df,
            numerical_cols=None,
            categorical_cols=None
    ):
        if numerical_cols is None:
            numerical_cols = ['num_reviews', 'review_score', 'total_positive', 'total_negative', 'price']
        if categorical_cols is None:
            categorical_cols = ['review_score_desc', 'developer', 'publisher', 'owners']

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine the numerical and categorical pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Apply the transformations to the games DataFrame

        game_features = preprocessor.fit_transform(df)
        game_nn_model = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(game_features)
        self.game_nn_model = game_nn_model
        self.game_name_df = df[['game_id', 'game_name']]


    def predict(self, player_id: int, num_recommendations: int=5):

        # Get the player's row in the CSR matrix
        if player_id not in self.user_index_mapping:
            print("User not found.")
            return []
        player_idx = self.user_index_mapping[player_id]

        # Calculate cosine similarity between this player and all other player
        player_similarities = cosine_similarity(
            self.player_matrix[player_idx].reshape(1, -1),
            self.player_matrix
        ).flatten()

        # Get the indices of the top similar users
        # first would be the player themselves
        top_similar_players_indices = player_similarities.argsort()[::-1][1:]

        # Aggregate the game preferences of similar users

        similar_players_game_interactions = self.user_game_matrix_csr[top_similar_players_indices]
        similar_players_game_scores = similar_players_game_interactions.sum(axis=0).A1

        # Use the KNN model to find games similar to those the user has reviewed positively
        player_interacted_indices = self.user_game_matrix_csr.getrow(player_idx).nonzero()[1]
        player_positive_game_indices = [
            i for i in player_interacted_indices if player_interacted_indices[player_idx, i] > 0
        ]
        # we need to normalize to make sure our weights can be added properly
        normalized_game_features = normalize(self.user_game_matrix_csr)
        knn_scores = np.zeros(normalized_game_features.shape[1])

        for game_idx in player_positive_game_indices:
            game_feature_vector = normalized_game_features.getrow(game_idx)
            distances, similar_game_indices = self.game_nn_model.kneighbors(
                game_feature_vector,
                n_neighbors=num_recommendations
            )

            for i, idx in enumerate(similar_game_indices.flatten()):
                # Inverse of distance can serve as a similarity score; avoid division by zero
                score = 1 / (1 + distances.flatten()[i])
                knn_scores[idx] += score

        svd_scores_normalized = similar_players_game_scores / np.max(similar_players_game_scores)
        knn_scores_normalized = knn_scores / np.max(knn_scores)
        combined_scores = (self.player_similarity_weight * svd_scores_normalized) + (self.game_similarity_weight * knn_scores_normalized)

        top_game_indices = combined_scores.argsort()[::-1][:num_recommendations]
        recommended_game_ids = [self.game_index_mapping[idx] for idx in top_game_indices]
        recommended_games = self.game_name_df[self.game_name_df['game_id'].isin(recommended_game_ids)]
        return recommended_games
