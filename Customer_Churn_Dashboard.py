import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Title and description
st.title('Customer Churn Predictor')
st.write('This is a predictive model integrated with a simple UI to highlight different customers’ likelihood to churn.')

# Header for user inputs
st.header('User Inputs')
name = st.text_input('Enter your name', 'John Doe')  # Default value is "John Doe"
age = st.slider('Select your age', 0, 100, 25)  # "Selected" corrected to "Select"
favourite_colour = st.selectbox("What's your favorite color?", ["Red", "Blue", "Green", "Yellow"])

# Display user inputs
st.write(f'Hello {name}, you are {age} years old, and your favorite color is {favourite_colour}.')

# Header for interactive chart
st.header('Interactive Chart')
num_points = st.slider('Number of points to plot', 10, 500, 100)
random_data = pd.DataFrame(
    np.random.randn(num_points, 2),
    columns=['X', 'Y']  # Ensure consistent column naming
)

# Scatterplot
st.write('Scatterplot of Random Data')
st.write('Use the slider above to adjust the number of points.')
fig, ax = plt.subplots()
ax.scatter(random_data['X'], random_data['Y'], alpha=0.6)
ax.set_title("Scatterplot of Random Points")  # Optional: Add a title for clarity
st.pyplot(fig)

# Header for random data table
st.header('Random Data Table')
st.write('Here is a preview of the random data:')
st.write(random_data.head())  # Display the first few rows of data
