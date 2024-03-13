from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from spam_filter_udf import spam_filter
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
import psycopg2
import pandas as pd
import os


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


cur = conn.cursor()
cur.execute("SELECT * FROM game_reviews")
rows = cur.fetchall()

# Convert to pandas DataFrame
df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])

# Convert pandas DataFrame to Spark DataFrame
sdf = spark.createDataFrame(df)

sdf.show()

spam_udf = udf(spam_filter,IntegerType())
processed_df = sdf.withColumn("is_spam", spam_udf(sdf["review"]))

processed_df.select("review", "is_spam") \
    .show(truncate=80, n=100)

spark.stop()
