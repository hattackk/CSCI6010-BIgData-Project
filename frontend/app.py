import streamlit as st
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sqlalchemy import create_engine, select, update, exc
import plotly.express as px
# database and api params
usr = os.environ.get('DB_USER')
pwd = os.environ.get("DB_PWD")
db_host = os.environ.get("DB_HOST")
db_port = os.environ.get("DB_PORT")
db_db = os.environ.get("DB_DATABASE")
DATABASE_URI = f'postgresql+psycopg2://{usr}:{pwd}@{db_host}:{db_port}/{db_db}'

engine = create_engine(DATABASE_URI)

def plot_review_reviews(engine):
    ## make a histogram of number of reviews per reviewer
    query = """
    select
    count(1) as nbr_of_reviews
    from game_reviews gr
    group by steamid
    """

    df1 = pd.read_sql(query, engine)
    plot1 = px.histogram(df1, x='nbr_of_reviews', nbins=200)
    plot1.update_layout(
        yaxis_type="log",
        title="Frequency of reviews",
        xaxis_title="Number of Reviewers",
        yaxis_title="Number of Reviews (Log Scale)"
    )
    st.plotly_chart(plot1, use_container_width=True)


def game_total_review_rating_headmap(engine):
    query = "select review_score, total_reviews from game_review_summary"
    df = pd.read_sql(query, engine)
    bin_edges = [0, 10, 100, 1000, 10000, 100000, 1000000]
    df['review_bins'] = pd.cut(
        df['total_reviews'],
        bins=bin_edges,
        labels=['0-10', '11-100', '101-1K', '1K-10K', '10K-100K', '100K-1M']
    )
    frequency_table_binned = pd.crosstab(df['review_score'], df['review_bins'])
    plot = go.Figure(data=go.Heatmap(
        x=frequency_table_binned.columns,
        y=frequency_table_binned.index,
        z=frequency_table_binned.values,
        # colorscale='Viridis',
        colorbar=dict(title='Frequency')
    ))

    # Update layout with titles
    plot.update_layout(
        title="2D Frequency Heatmap with Logarithmic X-Axis",
        xaxis_title="Total Reviews (Log Scale)",
        yaxis_title="Game Rating"
    )

    st.plotly_chart(plot, use_container_width=True)


st.header("Game Recommendation Engine")
st.header("", divider='rainbow')
tab1, tab2 = st.tabs(["Data Exploration", "Recommendations"])
with tab1:
    plot_review_reviews(engine)
    game_total_review_rating_headmap(engine)