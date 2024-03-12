from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType, IntegerType, BooleanType, FloatType
from spam_filter_udf import spam_filter
import generate_fake_review
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark import SparkFiles
import psycopg2
import pandas as pd
import os

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
    .config("spark.jars", "/postgresql-42.7.2.jar") \
    .config("spark.executor.extraClassPath", "/postgresql-42.7.2.jar") \
    .config("spark.driver.extraClassPath", "/postgresql-42.7.2.jar") \
    .getOrCreate() 
spark.sparkContext.addFile("/spam_filter_udf.py")
spark.sparkContext.addFile("/postgresql-42.7.2.jar")

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

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    dbname=db_db,
    user=usr,
    password=pwd,
    host=db_host,
    port=db_port
)

# Create a cursor object
cur = conn.cursor()

# Execute a query
cur.execute("SELECT * FROM game_reviews")

# Fetch all rows from the database
rows = cur.fetchall()

# Convert to pandas DataFrame
df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])

# Convert pandas DataFrame to Spark DataFrame
sdf = spark.createDataFrame(df)

sdf.show()

spam_udf = udf(spam_filter,IntegerType())
processed_df = sdf.withColumn("is_spam", spam_udf(sdf["review"]))

processed_df.select("review", "is_spam") \
    .show(truncate=False)

spark.stop()
