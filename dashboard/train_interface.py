import streamlit as st
from model_trainer import ModelTrainer

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
    model_name = st.text_input('Model Name',st.session_state.trainer.model_name)
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
