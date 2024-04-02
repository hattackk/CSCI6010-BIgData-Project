from pyspark.sql import SparkSession
from spam_filter_udf import spam_filter
from sentiment_analysis_udf import sentiment_analysis
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, FloatType
import psycopg2
import pandas as pd
from tqdm import tqdm
import os

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("GameReviewsPipeline") \
    .config("spark.jars", "/postgresql-42.7.2.jar") \
    .config("spark.executor.extraClassPath", "postgresql-42.7.2.jar") \
    .config("spark.driver.extraClassPath", "/postgresql-42.7.2.jar") \
    .getOrCreate() 
spark.sparkContext.addFile("/spam_filter_udf.py")
spark.sparkContext.addFile("/sentiment_analysis_udf.py")
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
properties = {
    "user": usr,
    "password": pwd,
    "driver": "org.postgresql.Driver"
}
DATABASE_URI = f"jdbc:postgresql://{db_host}:{db_port}/{db_db}"

cur = conn.cursor()
cur.execute("SELECT count(1) FROM game_reviews")
total_rows = cur.fetchall()[0][0]
chunk_size = 100

# define the spam filter
spam_udf = udf(spam_filter, IntegerType())
# define the sentiment analysis
sentiment_analysis_udf = udf(sentiment_analysis, FloatType())

# since we dont have a spark cluster, this will chunk the data
# into 1000 line segments which is more manageable locally
for offset in tqdm(range(0, total_rows, chunk_size)):
    query = f"(SELECT recommendationid, review FROM game_reviews LIMIT {chunk_size} OFFSET {offset}) AS chunk"
    chunk_df = spark.read.jdbc(url=DATABASE_URI, table=query, properties=properties)
    processed_df = chunk_df.withColumn("is_spam", spam_udf(chunk_df["review"]))
    sdf_with_sa = processed_df.withColumn("sentiment_score", sentiment_analysis_udf(processed_df["review"]))
    sdf_with_sa = sdf_with_sa.drop(sdf_with_sa.review)
    results = sdf_with_sa.collect()
    for row in results:
        cur.execute("""
            UPDATE game_reviews
            SET is_spam = %s, sentiment_score = %s
            WHERE recommendationid = %s
            """, (row.is_spam, row.sentiment_score, row.recommendationid))
    conn.commit()


spark.stop()
