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

def spider_chart(title, categories, lists_of_category_values):
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

def heatmap(data, title="Heatmap", cmap='viridis', annot=False, fmt=".2f"):
    """
    Create a heatmap for displaying in Streamlit.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame where each cell represents the heatmap intensity.
    title (str): Title of the heatmap.
    cmap (str): Colormap used for the heatmap. Check matplotlib colormaps for more options.
    annot (bool): If True, write the data value in each cell.
    fmt (str): String formatting code to use when adding annotations.
    """

    # Ensure the input is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")

    # Create the heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(data, cmap=cmap, annot=annot, fmt=fmt)
    plt.title(title)

    # Display the heatmap in Streamlit
    st.pyplot(plt)

def parallel_coordinates(df, dimensions, labels, sample_size=None, title='Parallel Coordinates Plot'):
    """
    Create and display a parallel coordinates chart in Streamlit.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    dimensions (list): List of column names to be used as dimensions in the plot.
    labels (dict): Dictionary mapping column names to labels for the plot.
    sample_size (int): Number of samples to visualize (default 20).
    title (str): Title of the plot.
    """

    # Sample the DataFrame if necessary
    if sample_size:
        df = df.sample(n=sample_size)

    fig = px.parallel_coordinates(df, dimensions=dimensions, labels=labels, 
                                  color_continuous_scale=px.colors.diverging.Tealrose, width=1200)
    fig.update_layout(
        title=title,
        xaxis_title='Categories',
        yaxis_title='Values', 
        autosize=True,
        margin=dict(l=100, r=100, t=80, b=40)
    )
    # Display the chart in Streamlit
    st.plotly_chart(fig)

def line_chart(df, dimensions, labels, title='Line Chart'):
    """
    Create and display a line chart in Streamlit.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    dimensions (list): List of column names to be used as dimensions in the plot.
    labels (dict): Dictionary mapping column names to labels for the plot.
    title (str): Title of the plot.
    """

    if df.empty or len(df) < 1:
        st.write("No data available to display the chart.")
        return

    # Ensuring the dimensions are in the DataFrame
    valid_dimensions = [dim for dim in dimensions if dim in df.columns]
    if not valid_dimensions:
        st.write("None of the specified dimensions are in the DataFrame.")
        return

    # Reshape DataFrame for the line chart
    reshaped_df = df[valid_dimensions].reset_index()
    reshaped_df = pd.melt(reshaped_df, id_vars=['index'], value_vars=valid_dimensions, 
                          var_name='Category', value_name='Value')

    # Replace column names with labels
    reshaped_df['Category'] = reshaped_df['Category'].map(labels)

    # Create the line chart
    fig = px.line(reshaped_df, x='Category', y='Value', color='index', markers=True)
    fig.update_layout(
        title=title,
        xaxis_title='Categories',
        yaxis_title='Values',
        autosize=True,
        margin=dict(l=100, r=100, t=80, b=40)
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)


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
    page_icon='🎮',
    layout='wide'
)

# dashboard title
st.title("CSCI 6010 Steam Analytics Dashboard")

users=users_df[users_df['steamid'].isin(game_reviews_df['steamid'].unique())].head(100) # only get users that have reviews
user_filter = st.selectbox("Select a user.", users)

user=users_df[users_df['steamid']==user_filter]
user

def userDisplay(user_filter):
    user_reviews=game_reviews_df[game_reviews_df['steamid']==user_filter]
    user_review_content = user_reviews['review']

    # Combine all reviews into one large text
    text = " ".join(review for review in user_review_content)

    wordcloud('Review Wordcloud', text)

    # Example usage
    categories = ['A', 'B', 'C', 'D', 'E']
    user1_values = [3, 5, 2, 4, 7]
    user2_values = [4, 3, 6, 5, 8]

    spider_chart("User Comparison", categories, [user1_values, user2_values])

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plt)
    with col2:
        data = pd.DataFrame(np.random.rand(10, 10), columns=[f'Col{i}' for i in range(1, 11)])
        heatmap(data, title="Most similar users", cmap='coolwarm', annot=True)

    dimensions = ['votes_up', 'votes_funny', 'weighted_vote_score', 'comment_count']
    labels = {'votes_up': 'Votes Up', 'votes_funny': 'Votes Funny', 'weighted_vote_score': 'Weighted Vote Score', 'comment_count': 'Comment Count'}

    line_chart(user_reviews, dimensions, labels, 'User Review Stats')


userDisplay(user_filter)

# get most similar users, mock for now
most_similar_users = users[users_df['steamid'] != user_filter].sample(10)
user_reviews = game_reviews_df[game_reviews_df['steamid'].isin(most_similar_users['steamid'])]
dimensions = ['votes_up', 'votes_funny', 'weighted_vote_score', 'comment_count']
labels = {'votes_up': 'Votes Up', 'votes_funny': 'Votes Funny', 'weighted_vote_score': 'Weighted Vote Score', 'comment_count': 'Comment Count'}

parallel_coordinates(user_reviews, dimensions, labels, title='Game Reviews - Parallel Coordinates Plot')

with st.form(key='similar_users'):
    # Use a selectbox for tab selection
    selected_steamid = st.radio("Top 10 most similar users:", most_similar_users['steamid'])
    button=st.form_submit_button('Inspect User.')

if button:
    with st.spinner('Analyzing user'):
        # Show content based on selected steamid
        user_info = users_df[users_df['steamid'] == selected_steamid]
        st.success('Analysis Ready.')

    st.write(f"SteamID: {selected_steamid} (User Similarity Rank {most_similar_users[most_similar_users['steamid']==selected_steamid].index})")
    userDisplay(selected_steamid)

