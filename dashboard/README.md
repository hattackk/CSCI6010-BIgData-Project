## Dashboard for Steam Game Comparison

## Steps

1. py -m venv venv
2. venv/Scripts/activate
3. pip install -r requirements.txt
4. Copy .env file from main project
5. py getData.py
6. streamlit run train_interface.py (for train dashboard)
6. streamlit run recommendation_dashboard.py (for user dashboard)

OR run train by itself (edit main function for test parameter, etc.):
py model_trainer.py