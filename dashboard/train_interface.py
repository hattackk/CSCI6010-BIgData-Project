import streamlit as st
from model_trainer import ModelTrainer

## BEGIN DASHBOARD LOGIC ##
st.set_page_config(
    page_title='Steam User Review Train Dashboard',
    page_icon='🎮',
    layout='wide'
)

# dashboard title
st.title("Steam Analytics Model Trainer")
trainer = ModelTrainer()


with st.form(key='train_params'):
    model_name = st.text_input('Model Name', trainer.model_name)
    col1, col2, col3 = st.columns(3)
    with col1:
        numerical_cols = st.multiselect("Choose numerical columns to train with  :", trainer.get_possible_numerical_cols())
    with col2:
        category_cols = st.multiselect("Choose categorical columns to train with  :", trainer.get_possible_categorical_cols())
    with col3:
        multi_cols = st.multiselect("Choose multi columns to train with  :", trainer.get_possible_multi_cols())
    

    test_enabled=st.checkbox('Perform test analysis?')
    test_count=st.slider('Test count:', 1,100)
    button=st.form_submit_button('Initiate Train.')

if button:
    with st.spinner('Analyzing user'):
        trainer.set_numerical_cols(numerical_cols)
        trainer.set_categorical_cols(category_cols)
        trainer.set_multi_cols(multi_cols)
        test = test_count if test_enabled else 0
        trainer.execute_train(test=test)
        st.success('Analysis Ready.')

        with st.form(key='test_model'):
            test_button=st.form_submit_button('MODEL TEST')

        if test_button:
            with st.spinner('Performing Test'):
                result = trainer.test_eval()
                print(result)
                result
