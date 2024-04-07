from __future__ import annotations
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
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


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


from typing import Any, Callable, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import check_is_fitted



class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Wraps `MultiLabelBinarizer` in a form that can work with `ColumnTransformer`. It makes it accept multiple inputs.

    Note that the input `X` has to be a `pandas.DataFrame`.
    """

    def __init__(self, binarizer_creator: Callable[[], Any] | None = None, dtype: npt.DTypeLike | None = None) -> None:
        self.binarizer_creator = binarizer_creator or MultiLabelBinarizer
        self.dtype = dtype

        self.binarizers = []
        self.categories_ = self.classes_ = []
        self.columns = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> MultiHotEncoder:  # noqa
        self.columns = X.columns.to_list()

        for column_name in X:
            print(column_name)
            binarizer = self.binarizer_creator().fit(X[column_name])
            self.binarizers.append(binarizer)
            self.classes_.append(binarizer.classes_)  # noqa

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)

        if len(self.classes_) != X.shape[1]:
            raise ValueError(f"The fit transformer deals with {len(self.classes_)} columns "
                             f"while the input has {X.shape[1]}.")

        return np.concatenate([binarizer.transform(X[c]).astype(self.dtype)
                               for c, binarizer in zip(X, self.binarizers)], axis=1)

    def get_feature_names_out(self, input_features: Sequence[str] = None) -> np.ndarray:
        check_is_fitted(self)

        cats = self.categories_

        if input_features is None:
            input_features = self.columns
        elif len(input_features) != len(self.categories_):
            raise ValueError(f"input_features should have length equal to number of features ({len(self.categories_)}),"
                             f" got {len(input_features)}")

        return np.asarray([input_features[i] + "_" + str(t) for i in range(len(cats)) for t in cats[i]])

class RecommenderModel:
    def __init__(self):
        self.game_nn_model = None
        self.player_matrix = None
        self.user_index_mapping = None
        self.game_index_mapping = None
        self.user_game_matrix_csr = None
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
        index_to_steamid = {i: steamid for steamid, i in user_index_mapping.items()}
        game_index_mapping = {index: game_id for index, game_id in enumerate(df['game_id'].cat.categories)}

        self.user_game_matrix_csr = user_game_matrix_csr
        self.user_index_mapping = user_index_mapping
        self.index_to_steamid = index_to_steamid
        self.game_index_mapping = game_index_mapping

    def train_game_similarities(
            self,
            df,
            numerical_cols=['num_reviews', 'review_score', 'total_positive', 'total_negative', 'price'],
            categorical_cols=['review_score_desc', 'developer', 'publisher', 'owners'],
            multi_label_cols=['categories', 'genres']
    ):
        transformers=[]
        
        if numerical_cols is not None and numerical_cols is not []:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, numerical_cols))

        if categorical_cols is not None and categorical_cols is not []:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_cols))

        if multi_label_cols is not None and multi_label_cols is not []:
            multilabel_transformer = Pipeline(steps=[
                ('onehot', MultiHotEncoder())
            ])
            transformers.append(('multi', multilabel_transformer, multi_label_cols))

        # Apply preprocessing pipelines
        preprocessor = ColumnTransformer(transformers)
        game_features = preprocessor.fit_transform(df)
        game_nn_model = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(game_features)
        self.game_nn_model = game_nn_model
        self.game_name_df = df[['game_id', 'game_name']]
        self.game_features = game_features


    def predict(self, player_id: int, num_recommendations: int = 5, player_similarity_filter=.9, player_similarity_weight=.5):
        player_idx = self.get_player_index(player_id)
        if player_idx is None:
            return []

        player_similarities = self.calculate_cosine_similarity(player_idx)
        top_similar_players_indices = self.get_most_similar_players_indices(player_similarities, player_similarity_filter)
        similar_players_game_scores = self.aggregate_game_preferences(top_similar_players_indices)
        knn_scores = self.find_similar_games_using_KNN(player_idx, num_recommendations)

        combined_scores = self.compute_combined_scores(similar_players_game_scores, knn_scores, player_similarity_weight)
        return self.get_top_recommendations(combined_scores, num_recommendations)


    def get_player_index(self, player_id):
        if player_id not in self.user_index_mapping:
            print("User not found.")
            return None
        return self.user_index_mapping[player_id]

    def calculate_cosine_similarity(self, player_idx):
        return cosine_similarity(
            self.user_game_matrix_csr[player_idx].reshape(1, -1),
            self.user_game_matrix_csr
        ).flatten()

    def get_most_similar_players_indices(self, player_similarities, player_similarity_filter):
        # Exclude exact matches (similarity of 1)
        player_similarities[player_similarities == 1.] = 0

        # Apply the similarity threshold
        eligible_indices = player_similarities >= player_similarity_filter

        # Get indices of players who meet the threshold
        top_similar_players_indices = player_similarities.argsort()[::-1][eligible_indices]
        return top_similar_players_indices

    def aggregate_game_preferences(self, top_similar_players_indices):
        similar_players_game_interactions = self.user_game_matrix_csr[top_similar_players_indices]
        return similar_players_game_interactions.sum(axis=0).A1

    def find_similar_games_using_KNN(self, player_idx, num_recommendations):
        player_interacted_indices = self.user_game_matrix_csr.getrow(player_idx).nonzero()[1]
        player_positive_game_indices = [
            i for i in player_interacted_indices if self.user_game_matrix_csr[player_idx, i] > 0
        ]
        normalized_game_features = normalize(self.game_features)
        knn_scores = np.zeros(normalized_game_features.shape[0])

        for game_idx in player_positive_game_indices:
            game_feature_vector = normalized_game_features.getrow(game_idx)
            distances, similar_game_indices = self.game_nn_model.kneighbors(
                game_feature_vector,
                n_neighbors=num_recommendations
            )

            for i, idx in enumerate(similar_game_indices.flatten()):
                distance = distances.flatten()[i]
                if np.isnan(distance):  # Check for NaN
                    print('NaNs!')
                    score = 0  # Handle NaN case, e.g., by setting score to 0
                else:
                    score = 1 / (1 + distance)
                knn_scores[idx] += score

        return knn_scores

    def compute_combined_scores(self, similar_players_game_scores, knn_scores, player_similarity_weight):
        # Transform negative scores to a positive dispreference scale
        transformed_player_scores = np.where(similar_players_game_scores > 0, 
                                            similar_players_game_scores, 
                                            1 + similar_players_game_scores)
        transformed_knn_scores = np.where(knn_scores > 0, 
                                        knn_scores, 
                                        1 + knn_scores)
        
        game_similarity_weight = 1 - player_similarity_weight

        # Combine scores with weights
        combined_scores = (player_similarity_weight * transformed_player_scores) + (game_similarity_weight * transformed_knn_scores)

        return combined_scores

    def get_top_recommendations(self, combined_scores, num_recommendations):
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
