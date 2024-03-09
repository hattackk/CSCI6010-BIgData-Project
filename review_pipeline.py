from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType, IntegerType, BooleanType, FloatType
import spam_filter_udf
import generate_fake_review
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("GameReviewsPipeline") \
    .getOrCreate()

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


# Load test data into DataFrame
df = spark.createDataFrame(test_data, schema=schema)

df.show()

spam_udf = udf(spam_filter_udf.spam_filter,IntegerType())

processed_df = df.withColumn("is_spam", spam_udf(df["review"]))

processed_df.show()

spark.stop()