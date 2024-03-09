import random
import faker

fake = faker.Faker()

def generate_review():
    '''
    Generates a fake review based on the game_review table schema
    '''
    return {
        "recommendationid": fake.random_number(digits=8),
        "steamid": fake.random_number(digits=10),
        "language": 'english',
        "review": fake.paragraph(),
        "timestamp_created": int(fake.date_time_this_century().timestamp()),
        "timestamp_updated": int(fake.date_time_this_century().timestamp()),
        "voted_up": random.choice([True, False]),
        "votes_up": fake.random_number(digits=4),
        "votes_funny": fake.random_number(digits=4),
        "weighted_vote_score": random.uniform(0, 1),
        "comment_count": fake.random_number(digits=2),
        "steam_purchase": random.choice([True, False]),
        "received_for_free": random.choice([True, False]),
        "written_during_early_access": random.choice([True, False]),
        "hidden_in_steam_china": random.choice([True, False]),
        "steam_china_location": fake.country_code(representation="alpha-2"),
        "application_id": fake.random_number(digits=5),
        "playtime_forever": fake.random_number(digits=4),
        "playtime_last_two_weeks": fake.random_number(digits=4),
        "playtime_at_review": fake.random_number(digits=4),
        "last_played": int(fake.date_time_this_century().timestamp())
    }