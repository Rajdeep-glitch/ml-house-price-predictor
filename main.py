import streamlit as st

import numpy as np
import pandas as pd
import plotly.express as px

def generate_house_data(n_samples=100):
    np.random.seed(42)
    # size = np.random.normal(1400, 50, n_samples)
    # price = size * 50 + np.random.normal(0, 50, n_samples)
    data = {
        'Square Footage': np.random.randint(500, 5000, n_samples),
        'Bedrooms': np.random.randint(1, 6, n_samples),
        'Bathrooms': np.random.randint(1, 4, n_samples),
        'Price': np.random.randint(100000, 1000000, n_samples)
    }
    df = pd.DataFrame(data)
    return df
    # return pd.DataFrame({'size': size, 'price': price})

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# def train_model():
def train_model(df):
    # df = generate_house_data(n_samples=100)
    # x = df[['size']]
    # y = df['price']
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) 
    x_train, x_test, y_train, y_test = train_test_split(df[['Square Footage', 'Bedrooms', 'Bathrooms']], df['Price'], test_size=0.2)

    model= LinearRegression()

    model.fit(x_train, y_train)

    return model

def main():
    st.title("House Price Prediction App")

    st.write("This app predicts house prices based on size.")

    # model= train_model()
    df = generate_house_data()
    model = train_model(df)

    # size= st.number_input('House Size', min_value=500, max_value=2000, value=1500)
    size = st.number_input('🏡 Square Footage', min_value=500, max_value=5000, value=1500)
    bedrooms = st.number_input('🛏️ Bedrooms', min_value=1, max_value=5, value=3)
    bathrooms = st.number_input('🛁 Bathrooms', min_value=1, max_value=3, value=2)

    if st.button('Predict Price'):
        # predicted_price = model.predict([[size]])
        predicted_price = model.predict([[size, bedrooms, bathrooms]])
        st.success(f'Estimated Price: Rs{predicted_price[0]:,.2f}')

        # df= generate_house_data()

        # fig = px.scatter(df, x='size', y='price', title='House Size vs Price')
        # fig.add_scatter(x=[size], y=[predicted_price[0]],
        #                 mode='markers',
        #                 marker= dict(color='red', size=15),
        #                 name='Predicted Price')
        fig = px.scatter(df, x='Square Footage', y='Price', 
                         size_max=10, title='House Size vs Price',
                         color='Bedrooms', hover_data=['Bathrooms'])
        
        fig.add_scatter(x=[size], y=[predicted_price[0]],
                        mode='markers',
                        marker=dict(color='red', size=15),
                        name='Predicted Price')

    
        
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
                        