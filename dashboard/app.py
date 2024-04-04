import streamlit as st
import numpy as np 
import pandas as pd
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx

from faker import Faker
import random

fake = Faker()

def subheader(text):
    st.markdown(f"### {text}")

def wordcloud(title, text):
    subheader(title)
    wordcloud = WordCloud(background_color="white", width=800, height=200).generate(text)

    plt.figure(figsize=(8,4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # Show the plot
    st.pyplot(plt)

def create_spider_chart(title, categories, lists_of_category_values):
    subheader(title)
    plt.clf()  # Clear the current figure

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)

    # Determine the max value across all datasets for setting y-ticks
    max_value = max(max(values) for values in lists_of_category_values)
    y_ticks = np.linspace(0, max_value, num=5)  # Creating 5 y-ticks
    y_tick_labels = [f"{tick:.2f}" for tick in y_ticks]
    plt.yticks(y_ticks, y_tick_labels, color="grey", size=7)
    plt.ylim(0, max_value)

    # Iterate over each list of values
    for idx, values in enumerate(lists_of_category_values):
        values = values + values[:1]  # Complete the loop
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'User {idx+1}')
        ax.fill(angles, values, alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

def load_dataframe_from_pickle(file_path):
    try:
        df = pd.read_pickle(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Assuming you have a list of table names
table_names = ["games", "game_rating", "game_review_summary", "game_reviews", "steam_users"]

dataframes = {}
for table_name in table_names:
    file_path = f"cache/{table_name}.pkl"
    df = load_dataframe_from_pickle(file_path)
    if df is not None:
        dataframes[table_name] = df

# DataFrames are now populated with random data generated using Faker
# games_df, game_rating_df, game_review_summary_df, game_reviews_df, steam_users_df

users_df=dataframes['steam_users']
game_reviews_df=dataframes['game_reviews']
print(f"total reviews == {len(game_reviews_df)}")
game_review_summary_df=dataframes['game_review_summary']
game_rating_df=dataframes['game_rating']
games_df=dataframes['games']

## BEGIN DASHBOARD LOGIC ##
st.set_page_config(
    page_title='Steam User Review Analysis',
    page_icon='✅',
    layout='wide'
)

# dashboard title
st.title("CSCI 6010 Steam Analytics Dashboard")

users=users_df[users_df['steamid'].isin(game_reviews_df['steamid'].unique())].head()
user_filter = st.selectbox("Select a user.", users)

user=users_df[users_df['steamid']==user_filter]
user

print("User vocabulary:")
user_reviews=game_reviews_df[game_reviews_df['steamid']==user_filter]['review']
print(user_reviews)
# Combine all reviews into one large text
text = " ".join(review for review in user_reviews)

wordcloud('Review Wordcloud', text)

# Example usage
categories = ['A', 'B', 'C', 'D', 'E']
user1_values = [3, 5, 2, 4, 7]
user2_values = [4, 3, 6, 5, 8]

create_spider_chart("User Comparison", categories, [user1_values, user2_values])
plt.show()


# create_spider_chart()
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plt)
with col2:
    ## PARALLEL COORDINATES
    filtered_reviews_df = game_reviews_df.sample(n=20)  # Example: taking a sample for visualization

    # Create a parallel coordinates chart
    fig = px.parallel_coordinates(filtered_reviews_df, 
                                dimensions=['votes_up', 'votes_funny', 'weighted_vote_score', 'comment_count'],
                                labels={'votes_up': 'Votes Up', 
                                        'votes_funny': 'Votes Funny', 
                                        'weighted_vote_score': 'Weighted Vote Score', 
                                        'comment_count': 'Comment Count'},
                                color_continuous_scale=px.colors.diverging.Tealrose, 
                                width=800)
    fig.update_layout(
        title='Parallel Coordinates Plot',
        xaxis_title='Sentiments',
        yaxis_title='Values'  # Adjust left, right, top, bottom margins as needed
    )


    # Display the chart in Streamlit
    st.plotly_chart(fig)