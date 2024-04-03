from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import os
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import dill as pickle
import lzma


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
        self.game_features = None

    def train_user_similaries(self, df, n_components=50):
        """
        trains the TruncatedSVD matrix for user similarities based on sentiment score
        :param df: dataframe with steamid, application_id, sentiment_score
        :return: None
        """
        df = df.copy()
        df = df.sort_values(by='application_id')
        # use a sparse matrix representation of the matrix
        df['steamid'] = df['steamid'].astype("category")
        df['game_id'] = df['application_id'].astype("category")

        row_ind = df['steamid'].cat.codes
        col_ind = df['game_id'].cat.codes

        user_game_matrix_sparse = coo_matrix(
            (df['voted_up'], (row_ind, col_ind)),
            shape=(df['steamid'].nunique(), df['game_id'].nunique())
        )
        user_game_matrix_csr = user_game_matrix_sparse.tocsr()
        user_index_mapping = {steamid: index for index, steamid in enumerate(df['steamid'].cat.categories)}
        game_index_mapping = {index: game_id for index, game_id in enumerate(df['game_id'].cat.categories)}

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
        self.game_features = game_features


    def predict(self, player_id: int, num_recommendations: int=5):

        # Get the player's row in the CSR matrix
        if player_id not in self.user_index_mapping:
            print("User not found.")
            return []
        player_idx = self.user_index_mapping[player_id]

        # Calculate cosine similarity between this player and all other player
        player_similarities = cosine_similarity(
            self.user_game_matrix_csr[player_idx].reshape(1, -1),
            self.user_game_matrix_csr
        ).flatten()

        # filter out exact matches since those players will not give useful reviews
        nbr_exact = len(player_similarities[player_similarities == 1.])

        top_similar_players_indices = player_similarities.argsort()[::-1][nbr_exact:]

        # Aggregate the game preferences of similar users
        similar_players_game_interactions = self.user_game_matrix_csr[top_similar_players_indices]
        similar_players_game_scores = similar_players_game_interactions.sum(axis=0).A1

        # Use the KNN model to find games similar to those the user has reviewed positively
        player_interacted_indices = self.user_game_matrix_csr.getrow(player_idx).nonzero()[1]
        player_positive_game_indices = [
            i for i in player_interacted_indices if self.user_game_matrix_csr[player_idx, i] > 0
        ]
        # we need to normalize to make sure our weights can be added properly
        normalized_game_features = normalize(self.game_features)
        knn_scores = np.zeros(normalized_game_features.shape[0])

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

        cosine_scores_normalized = similar_players_game_scores / np.max(similar_players_game_scores)
        knn_scores_normalized = knn_scores / np.max(knn_scores)
        combined_scores = (self.player_similarity_weight * cosine_scores_normalized) + (self.game_similarity_weight * knn_scores_normalized)

        top_game_indices = combined_scores.argsort()[::-1][:num_recommendations]
        recommended_game_ids = [self.game_index_mapping[idx] for idx in top_game_indices]
        recommended_games = self.game_name_df[self.game_name_df['game_id'].isin(recommended_game_ids)]
        return recommended_games

    def save(self, model_name: str):
        pickled_model = pickle.dumps(self)
        with lzma.open(model_name, "wb") as f:
            f.write(pickled_model)

    @staticmethod
    def load(location: str):
        with lzma.open(location, "rb") as f:
            model = pickle.loads(f.read())
        return model
