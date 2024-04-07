import streamlit as st
import numpy as np 
import pandas as pd
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx

from model import RecommenderModel
from numpy.random import randint
from tabulate import tabulate

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
    st.pyplot(plt)

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

def adjusted_heatmap(data, title="Heatmap", cmap='viridis', annot=False, fmt=".2f"):
    """
    Create a heatmap for displaying in Streamlit.
    ...
    """

    # Set a large figure size to accommodate all genres
    plt.figure(figsize=(10, 8))  # Adjust the size to your needs

    # Rotate x-axis labels if necessary
    ax = sns.heatmap(data, cmap=cmap, annot=annot, fmt=fmt)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Show only annotations for values above a certain threshold
    if annot:
        for text in ax.texts:
            t = float(text.get_text())
            if abs(t) < -1.0:  # Set a threshold value that makes sense for your data
                text.set_text('')

    plt.title(title)
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

def bar_chart(data, category_col, value_col, chart_title='Bar Chart', x_label='Category', y_label='Value', rotation_angle=90, figsize=(10, 6)):
    """
    Plots a bar chart with bars colored red for negative values and green for positive values.

    :param data: pandas DataFrame containing the data.
    :param category_col: String name of the column containing the category names.
    :param value_col: String name of the column containing the numerical values.
    :param chart_title: Title of the chart (default 'Bar Chart').
    :param x_label: Label for the x-axis (default 'Category').
    :param y_label: Label for the y-axis (default 'Value').
    :param rotation_angle: Rotation angle for x-axis labels (default 90).
    :param figsize: Tuple for the figure size (default (10, 6)).
    """
    plt.figure(figsize=figsize)
    bar_colors = ['green' if x > 0 else 'red' for x in data[value_col]]
    plt.bar(data[category_col], data[value_col], color=bar_colors)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(chart_title)
    plt.xticks(rotation=rotation_angle)
    st.pyplot(plt)

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

table_names = ["games", "game_rating", "game_review_summary", "game_reviews", "steam_users", "app_type"]

dataframes = {}
for table_name in table_names:
    file_path = f"cache/{table_name}.pkl"
    df = load_dataframe_from_pickle(file_path)
    if df is not None:
        dataframes[table_name] = df

st.session_state.users_df=dataframes['steam_users']
st.session_state.game_reviews_df=dataframes['game_reviews']
print(f"total reviews == {len(st.session_state.game_reviews_df)}")
st.session_state.game_review_summary_df=dataframes['game_review_summary']
st.session_state.game_rating_df=dataframes['game_rating']
st.session_state.games_df=dataframes['games']
st.session_state.types_df=dataframes['app_type']
st.session_state.games_df = pd.merge(st.session_state.games_df, st.session_state.types_df, left_on='game_id', right_on='app_id', how='left')
st.session_state.game_reviews_df = pd.merge(st.session_state.game_reviews_df, st.session_state.games_df, left_on='application_id', right_on='game_id', how='left')

def get_user_sentiment_per_genre(user_filter):
    user_reviews=st.session_state.game_reviews_df[st.session_state.game_reviews_df['steamid']==user_filter]
    # Explode 'genres' into separate rows
    genres_exploded = user_reviews[['genres', 'sentiment_score']].explode('genres')

    # Explode 'categories' into separate rows
    categories_exploded = user_reviews[['categories', 'sentiment_score']].explode('categories')

    # Concatenate the exploded DataFrames
    combined_exploded = pd.concat([genres_exploded, categories_exploded.rename(columns={'categories': 'genres'})])

    # Group by each item in 'genres' (which now includes categories) and calculate the average sentiment score
    return combined_exploded.groupby('genres')['sentiment_score'].mean().reset_index()

def user_display(user_filter):
    user_reviews=st.session_state.game_reviews_df[st.session_state.game_reviews_df['steamid']==user_filter]
    avg_sentiment= get_user_sentiment_per_genre(user_filter)


    bar_chart(avg_sentiment, 'genres', 'sentiment_score', chart_title='Average Sentiment Score by Genre/Category', x_label='Genres/Categories', y_label='Average Sentiment Score')
    user_review_content = user_reviews['review']

    # Combine all reviews into one large text
    text = " ".join(review for review in user_review_content)

    wordcloud('Review Wordcloud', text)
    st.session_state.user_categories_selected = 4 # set default

    with st.form('review_stats'):
        num_categories=st.slider('Number of Categories:', 2, 10, st.session_state.user_categories_selected)
        categories_selected = st.form_submit_button('Reanalyze Categories')

    if categories_selected:
        st.session_state.user_categories_selected = num_categories
    

    st.markdown(f'#### User `{user_filter}` Top {st.session_state.user_categories_selected} categories.')
    topX = avg_sentiment.sort_values('sentiment_score', ascending=False).head(st.session_state.user_categories_selected)
    topX

    charts = ['votes_up', 'votes_funny', 'weighted_vote_score', 'comment_count'] # 1 chart each with these as Y axis
    cats  = topX['genres'].values # these are the x axes on each chart
    labels = {'votes_up': 'Votes Up', 'votes_funny': 'Votes Funny', 'weighted_vote_score': 'Weighted Vote Score', 'comment_count': 'Comment Count'}

    # now we have top X categories, need to go back and filter user_reviews df for rows with those
    top_category_reviews = user_reviews[user_reviews.apply(lambda row: any(item in cats for item in row['categories']) or
                                         any(item in cats for item in row['genres']), axis=1)]
    
    top_category_reviews

    cols = st.columns(len(charts))
    for i, col in enumerate(cols):
        with col:
            chart = charts[i]
            plot_data = {}
            for target in cats:
                # Sum values for each genre/category
                sum_value = top_category_reviews[top_category_reviews.apply(lambda row: target in row['genres'] or target in row['categories'], axis=1)][chart].sum()
                plot_data[target] = sum_value

            # Convert plot_data to a DataFrame
            plot_df = pd.DataFrame(list(plot_data.items()), columns=['Genre/Category', chart])

            # Creating the line plot
            plt.figure(figsize=(10, 6))
            plt.plot(plot_df['Genre/Category'], plot_df[chart], marker='o')

            # Adding title and labels
            plt.title(f'Sum of {labels[chart]} per Genre/Category')
            plt.xlabel('Genre/Category')
            plt.ylabel(labels[chart])

            st.pyplot(plt)

def user_recommendations(user_filter):
    st.markdown('# User Recommendations')
    user_reviews=st.session_state.game_reviews_df[st.session_state.game_reviews_df['steamid']==user_filter]
    st.session_state.top_similar_players_indices = None
    with st.form(key='recommendations'):
        c1,c2 = st.columns(2)
        with c1:
            model_name=st.text_input('Model name','test.pkl.xz')
            similarity_weight = st.slider('User similarity weighting (game weight will be inverse)', 0.0,1.0,.5)
            similarity_filter = st.slider('User similarity filter', -1.0,1.0,.85)
        with c2:
            model_button=st.form_submit_button('Get recommendations.')

        if model_button:
            model = RecommenderModel.load(f'./{model_name}')
            users = list(model.user_index_mapping.keys())
            player_id = user_filter
            num_recommendations = 5
            print(dir(model))
            player_idx = model.get_player_index(player_id) 
            if player_idx is None:
                st.markdown("## Player not found in model :(")

            player_similarities = model.calculate_cosine_similarity(player_idx)
            top_similar_players_indices = model.get_most_similar_players_indices(player_similarities, similarity_filter)
            similar_players_game_scores = model.aggregate_game_preferences(top_similar_players_indices)
            knn_scores = model.find_similar_games_using_KNN(player_idx, num_recommendations)

            combined_scores = model.compute_combined_scores(similar_players_game_scores, knn_scores, similarity_weight)
            recommendations = model.get_top_recommendations(combined_scores, num_recommendations)
            recommendations
            
            
            st.session_state.top_similar_players_indices = top_similar_players_indices
            f"Similar users found: {len(top_similar_players_indices)}" 

    if st.session_state.top_similar_players_indices is not None:
        steamids = [model.index_to_steamid[index] for index in st.session_state.top_similar_players_indices]
        most_similar_users = st.session_state.users_df[st.session_state.users_df['steamid'].isin(steamids)]

        user_reviews = st.session_state.game_reviews_df[st.session_state.game_reviews_df['steamid'].isin(most_similar_users['steamid'])]
        
        user_reviews

        user_genre_sentiment_dfs = []
        for user in steamids:
            user_sentiment_df = get_user_sentiment_per_genre(user)
            user_sentiment_df['user'] = user
            user_genre_sentiment_dfs.append(user_sentiment_df)
        # Concatenate all user DataFrames
        combined_df = pd.concat(user_genre_sentiment_dfs)
        heatmap_df = combined_df.pivot_table(index='user', columns='genres', values='sentiment_score', fill_value=0)
        
        col1, col2 = st.columns(2)
        with col1:
            selected_users_df = heatmap_df.iloc[:5]
            categories = list(selected_users_df.columns)
            lists_of_category_values = selected_users_df.values.tolist()
            title = "User Comparison by Genre Sentiment"
            spider_chart(title, categories, lists_of_category_values)
        with col2:
            heatmap(heatmap_df, title="Most similar users", cmap='coolwarm')

           
        c1,c2 = st.columns(2)
        with c1:
            pivot_table = user_reviews.pivot_table(values='sentiment_score', index='game_name', columns='steamid', fill_value=0)
            heatmap(pivot_table, title="User Sentiment per game.")
        with c2:
            dimensions = ['votes_up', 'votes_funny', 'weighted_vote_score', 'comment_count']
            labels = {'votes_up': 'Votes Up', 'votes_funny': 'Votes Funny', 'weighted_vote_score': 'Weighted Vote Score', 'comment_count': 'Comment Count'}
            line_chart(user_reviews, dimensions, labels, 'User Review Stats')

        with st.form(key='similar_users'):
            selected_steamid = st.radio("Top 10 most similar users:", most_similar_users['steamid'])
            button=st.form_submit_button('Inspect User.')

        if button:
            with st.spinner('Analyzing user'):
                user_info = st.session_state.users_df[st.session_state.users_df['steamid'] == selected_steamid]
                st.success('Analysis Ready.')

            st.write(f"SteamID: {selected_steamid} (User Similarity Rank {most_similar_users[most_similar_users['steamid']==selected_steamid].index})")
            user_display(selected_steamid)

## BEGIN DASHBOARD LOGIC ##
st.set_page_config(
    page_title='Steam User Review Analysis',
    page_icon='🎮',
    layout='wide'
)

# dashboard title
st.title("CSCI 6010 Steam Analytics Dashboard")

users=st.session_state.users_df[st.session_state.users_df['steamid'].isin(st.session_state.game_reviews_df['steamid'].unique())].sort_values(by='num_reviews', ascending=False).head(100) # only get users that have reviews
user_filter = st.selectbox("Select a user.", users)
user=users[users['steamid']==user_filter]
user

user_display(user_filter)
user_recommendations(user_filter) 

#  Streamlit form types:
# Radio Buttons (st.radio): Good for selecting one option from a small set.
# Dropdown Menu (st.selectbox): Useful for selecting one option from a larger set.
# Multi-select Dropdown (st.multiselect): Allows selection of multiple options from a dropdown list.
# Checkboxes (st.checkbox): For toggle options.
# Slider (st.slider): To select a value from a range.
# Text Input (st.text_input): For text input.
# Number Input (st.number_input): For numerical input.
# Date Input (st.date_input): To input dates.
# Time Input (st.time_input): For inputting time values.
# File Uploader (st.file_uploader): To upload files.
# Buttons (st.button): Regular clickable buttons.