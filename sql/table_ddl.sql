-- DROP TABLE games, game_rating, game_review_summary, game_reviews, steam_users;

-- Creating table 'games'
CREATE TABLE games (
    game_id SERIAL PRIMARY KEY,
    game_name VARCHAR NOT NULL,
    developer VARCHAR,
    publisher VARCHAR,
    owners VARCHAR,
    price FLOAT,
    initialprice FLOAT,
    discount FLOAT,
    ccu INTEGER
);

-- Creating table 'game_rating'
CREATE TABLE game_rating (
    game_id INTEGER PRIMARY KEY REFERENCES games(game_id),
    score_rank INTEGER,
    positive INTEGER,
    negative INTEGER,
    userscore INTEGER,
    average_forever INTEGER,
    average_2weeks INTEGER,
    median_forever INTEGER,
    median_2weeks INTEGER
);

-- Creating table 'game_review_summary'
CREATE TABLE game_review_summary (
    game_id INTEGER PRIMARY KEY REFERENCES games(game_id),
    num_reviews INTEGER,
    review_score INTEGER,
    review_score_desc VARCHAR,
    total_positive INTEGER,
    total_negative INTEGER,
    total_reviews INTEGER
);

-- Creating table 'game_reviews'
CREATE TABLE game_reviews (
    recommendationid BIGINT PRIMARY KEY,
    author BIGINT,
    language VARCHAR,
    review VARCHAR,
    timestamp_created INT,
    timestamp_updated INT,
    voted_up BOOLEAN,
    votes_up BIGINT,
    votes_funny BIGINT,
    weighted_vote_score FLOAT,
    comment_count INT,
    steam_purchase BOOLEAN,
    received_for_free BOOLEAN,
    written_during_early_access BOOLEAN,
    hidden_in_steam_china BOOLEAN,
    steam_china_location VARCHAR,
    application_id INT REFERENCES games(game_id)
);

-- Creating table 'steam_users'
CREATE TABLE steam_users (
    steamid BIGINT PRIMARY KEY,
    num_games_owned INT,
    num_reviews INT,
    playtime_forever INT,
    playtime_last_two_weeks INT,
    playtime_at_review INT,
    last_played INT
);

-- Creating table 'game_review_download_status
CREATE TABLE game_review_download_status (
  game_id INTEGER PRIMARY KEY REFERENCES games(game_id),
  status varchar
);

-- Foreign key from 'game_reviews' to 'steam_users'
ALTER TABLE game_reviews ADD CONSTRAINT fk_game_reviews_author FOREIGN KEY (author) REFERENCES steam_users(steamid);