import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor


header=st.beta_container()
dataset=st.beta_container()
model_training=st.beta_container()

st.markdown(
    """
    <style>
   .main{
    background-color: #F5F5F5;
    }
    </style>
   """,
   unsafe_allow_html=True
)

@st.cache
def get_data(filename):
    salaries=pd.read_csv(filename)

    return salaries


with header:
    st.title('Salary Prediction Data Science App')
    st.text('In this project i look into position of employees and their salaries in the company')

with dataset:
    st.header('Position-Salary Dataset')
    st.text('I found this dataset on superdatascience.com')

    salary=get_data('DATA/Position_Salaries.csv')
    st.write(salary.head())


with model_training:
    st.header('Its time to train the model')
    st.text('Here you get to choose the hyper parameters of the model and see how the performance changes')

    sel_col,dis_col=st.beta_columns(2)

    random_state=sel_col.slider('what should be the random state of the model?',min_value=0,max_value=50,value=0,step=5)
    n_estimators=sel_col.selectbox('How many trees should be there?',options=[5,10,15,20,25,50,100,125,150,200,250,300,'No limit'],index=0)

    sel_col.text('Here is a list of input features in my data')
    sel_col.write(salary.columns)



    if n_estimators=='No limit':
        regr = RandomForestRegressor(random_state=random_state)
    else:
        regr=RandomForestRegressor(random_state=random_state,n_estimators=n_estimators)

    x=salary.iloc[:,1:-1].values
    y=salary[['Salary']]


    regr.fit(x,y)
    p=sel_col.text_input('Enter a value to predict',6.5)
    prediction=regr.predict([[p]])
    prediction2=regr.predict(y)

    dis_col.subheader('The Predicted value is:')
    dis_col.text(prediction)

    dis_col.subheader('Visualising The Random Forest Regression Result')

    X_grid = np.arange(min(x), max(x), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(x, y, color='red')
    plt.plot(X_grid, regr.predict(X_grid), color='blue')
    plt.title('Truth or Bluff (Random Forest Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
