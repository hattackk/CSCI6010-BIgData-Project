from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType, IntegerType, BooleanType, FloatType
from spam_filter_udf import spam_filter
import generate_fake_review
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark import SparkFiles

# import json
# import os
# import time
# import logging
# import click
# from dotenv import load_dotenv
# from sqlalchemy import create_engine, select, update, exc

# from database_tables import (
#     metadata,
#     game_review_summary_table, steam_users_table, game_reviews_table, game_review_download_status_table
# )
# from load_json_to_database import parse_single_game_review, parse_game_json, add_or_update
# from steam_api_client import SteamAPIClient

# load_dotenv()
# logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # database and api params
# usr = os.environ.get('DB_USER')
# pwd = os.environ.get("DB_PWD")
# db_host = os.environ.get("DB_HOST")
# db_port = os.environ.get("DB_PORT")
# db_db = os.environ.get("DB_DATABASE")
# DATABASE_URI = f'postgresql+psycopg2://{usr}:{pwd}@{db_host}:{db_port}/{db_db}'


# Define schema for game_reviews table
schema = StructType([
    StructField("recommendationid", LongType(), nullable=False),
    StructField("steamid", LongType(), nullable=False),
    StructField("language", StringType(), nullable=False),
    StructField("review", StringType(), nullable=False),
    StructField("timestamp_created", IntegerType(), nullable=False),
    StructField("timestamp_updated", IntegerType(), nullable=False),
    StructField("voted_up", BooleanType(), nullable=False),
    StructField("votes_up", LongType(), nullable=False),
    StructField("votes_funny", LongType(), nullable=False),
    StructField("weighted_vote_score", FloatType(), nullable=False),
    StructField("comment_count", IntegerType(), nullable=False),
    StructField("steam_purchase", BooleanType(), nullable=False),
    StructField("received_for_free", BooleanType(), nullable=False),
    StructField("written_during_early_access", BooleanType(), nullable=False),
    StructField("hidden_in_steam_china", BooleanType(), nullable=False),
    StructField("steam_china_location", StringType(), nullable=False),
    StructField("application_id", IntegerType(), nullable=False),
    StructField("playtime_forever", IntegerType(), nullable=False),
    StructField("playtime_last_two_weeks", IntegerType(), nullable=False),
    StructField("playtime_at_review", IntegerType(), nullable=False),
    StructField("last_played", IntegerType(), nullable=False)
])

# Generate test data for 5 reviews
test_data = [generate_fake_review.generate_review() for _ in range(5)]

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("GameReviewsPipeline") \
    .getOrCreate() 
spark.sparkContext.addFile("/spam_filter_udf.py")

# Load test data into DataFrame
df = spark.createDataFrame(test_data, schema=schema)

df.show()

spam_udf = udf(spam_filter,IntegerType())

processed_df = df.withColumn("is_spam", spam_udf(df["review"]))

processed_df.show()

spark.stop()