import streamlit as st
from model_trainer import ModelTrainer
import pandas as pd

# Initialize session state variables
if 'trainer' not in st.session_state:
    st.session_state.trainer = ModelTrainer()

if 'analysis_ready' not in st.session_state:
    st.session_state.analysis_ready = False

if 'test_result' not in st.session_state:
    st.session_state.test_result = None

## BEGIN DASHBOARD LOGIC ##
st.set_page_config(
    page_title='Steam User Review Train Dashboard',
    page_icon='🎮',
    layout='wide'
)

# dashboard title
st.title("Steam Analytics Model Trainer")


with st.form(key='train_params'):
    ncol1,ncol2 = st.columns(2)
    with ncol1:
        model_name = st.text_input('Model Name',st.session_state.trainer.model_name)
    with ncol2:
        inner_col1, inner_col2, inner_col3 = st.columns(3)
        users_df = pd.read_pickle(f"./cache/steam_users.pkl")
        app_type_df = pd.read_pickle(f"./cache/app_type.pkl")
        review_df = pd.read_pickle(f"./cache/game_reviews.pkl")
   
        total_users = len(users_df)
        total_reviews = len(review_df)
        total_games = len(app_type_df)
        unique_reviews = 0
        unique_games = 0
        genres_list = app_type_df['genres'].explode()

        # Flatten the arrays in the 'categories' column
        categories_list = app_type_df['categories'].explode()

        # Count unique values in 'genres' and 'categories'
        unique_genres_count = genres_list.nunique()
        unique_categories_count = categories_list.nunique()

        # Create strings
        unique_genres_string = f"Unique Genres: {unique_genres_count}"
        unique_categories_string = f"Unique Categories: {unique_categories_count}"
        # Create markdown text
        with inner_col1:
            markdown_text = f"""
                - Unique Genres: {unique_genres_count:,}\n
                - Unique Categories: {unique_categories_count:,}\n
            """
            st.markdown(markdown_text)
        with inner_col2:
            markdown_text = f"""
                - Unique Games: {total_games:,}\n
                - Unique Users: {total_users:,}\n
            """
            st.markdown(markdown_text)
        with inner_col3:
            markdown_text = f"""
                - Unique Reviews: {total_reviews:,}\n
            """
            st.markdown(markdown_text)





    col1, col2, col3 = st.columns(3)
    with col1:
        numerical_cols = st.multiselect("Choose numerical columns to train with :", st.session_state.trainer.get_possible_numerical_cols(), st.session_state.trainer.get_possible_numerical_cols())
    with col2:
        category_cols = st.multiselect("Choose categorical columns to train with :", st.session_state.trainer.get_possible_categorical_cols(), st.session_state.trainer.get_possible_categorical_cols())
    with col3:
        multi_cols = st.multiselect("Choose multi columns to train with :", st.session_state.trainer.get_possible_multi_cols(), st.session_state.trainer.get_possible_multi_cols())
    
    test_enabled=st.checkbox('Perform test analysis?')
    test_count=st.slider('Test count:', 1,100)
    button=st.form_submit_button('Initiate Train.')
if button:
    with st.spinner('Analyzing user'):
        st.session_state.trainer.set_numerical_cols(numerical_cols)
        st.session_state.trainer.set_categorical_cols(category_cols)
        st.session_state.trainer.set_multi_cols(multi_cols)
        st.session_state.trainer.set_model_name(model_name)
        test = test_count if test_enabled else 0
        st.session_state.trainer.execute_train(test=test)
        st.success('Analysis Ready.')
        st.session_state.analysis_ready=True

if st.session_state.analysis_ready:
    with st.form(key='test_model'):
        c1,c2,_,_,_ = st.columns(5)
        with c1:
            st.markdown("### Model Trained")
        with c2:
           test_button=st.form_submit_button('RUN MODEL TEST')

    if test_button:
        with st.spinner('Performing Test'):
            st.session_state.test_result = st.session_state.trainer.test_eval()

            if st.session_state.test_result is not None:
                df=st.session_state.test_result
                st.session_state.test_result.head()
                df.head()
                columns = list(df.columns)

                # Move the last column to be the first
                columns = [columns[-1]] + columns[:-1]

                # Reorder the DataFrame columns
                df_reordered = df[columns]

                df_reordered
