from model import RecommenderModel
from numpy.random import randint
from tabulate import tabulate
model = RecommenderModel.load('./test.pkl.xz')
users = list(model.user_index_mapping.keys())
user_id = users[randint(0, len(users))]
print(f'user id: {users[randint(0, len(users))]}')
recommendation = model.predict(user_id)
print(tabulate(recommendation, showindex=False, headers='keys', tablefmt='psql'))