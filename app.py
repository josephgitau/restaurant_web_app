# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestRegressor

# Title
st.title('Restaurant Price Prediction Web App') 
st.markdown('''
            This web app is created to predict the price of a restaurant based on various features. 
            The dataset is taken from Kaggle. The dataset contains 11790 rows and 4 columns. 
            The dataset contains the following columns:

            - item_name
            - category
            - price
            - hour
            - day
            ''')

# Reading the dataset
df = pd.read_csv('data.csv', sep='|')
features= ['item_name', 'category', 'hour', 'day']
# extract hour of the day from order_time
df['order_time'] = pd.to_datetime(df['order_time'])
df['hour'] = df['order_time'].dt.hour

# extract day of the week from order_date
df['order_date'] = pd.to_datetime(df['order_date'])
df['day'] = df['order_date'].dt.day_name()

st.subheader('Dataset sample')
st.dataframe(df[features].head())

# We get inputs from users 
st.header('User Input Features: ')
item_name = st.selectbox('Item Name', df['item_name'].unique())
category = st.selectbox('Category', df['category'].unique())
hour = st.slider('Hour', 0, 23, 12)
day = st.selectbox('Day', df['day'].unique())

# We create a dictionary from the inputs
user_input = {'item_name': item_name,
              'category': category,
              'hour': hour,
              'day': day}

# Transforming user input into a dataframe
input_df = pd.DataFrame([user_input])

# display the user input
st.subheader('User Input Features')
st.dataframe(input_df)

# import my encoder
import pickle
encoder = pickle.load(open('encoder.pkl', 'rb'))


# import our model
model = pickle.load(open('model.pkl', 'rb'))

# Encoding the inputs
input_df = pd.DataFrame(encoder.transform(input_df), columns=features)
# display the encoded features
st.subheader('Encoded User Input Features')
st.dataframe(input_df)

# Predicting the price
price = model.predict(input_df)
st.subheader('Predicted Price')
st.write('The predicted price is: ', price[0])
