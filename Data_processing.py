import numpy as np
import pandas as pd

# Generate basic customer data
n = 1000000
age = np.random.randint(18, 70, n)
gender = np.random.choice([0, 1], n, p=[0.5, 0.5])
loyalty = np.random.choice([0, 1], n, p=[0.7, 0.3])
income = np.random.normal(50000, 15000, n).clip(20000, 120000)


# Generate product data
categories = ['bags & luggage', 'stores', 'home & kitchen', 'grocery & gourmet foods',
 'sports & fitness', "kids' fashion", "women's shoes", "women's clothing",
 "men's clothing", "men's shoes", 'car & motorbike', 'accessories',
 'appliances', 'tv, audio & cameras', 'beauty & health',
 'industrial supplies', 'toys & baby products', 'pet supplies', 'music',
 'home, kitchen, pets']


probabilities = [0.05, 0.03, 0.06, 0.06, 0.07, 0.07, 0.08, 0.05, 
                 0.02, 0.01, 0.04, 0.06, 0.08, 0.03, 0.08, 0.02, 
                 0.07, 0.02, 0.03, 0.07]

product_category = np.random.choice(categories, n, p=probabilities)
product_price = np.random.normal(100, 50, n).clip(5, 500)
discount = np.random.choice([0, 1], n, p=[0.6, 0.4])

# Simulate behavior
time_spent = np.random.lognormal(mean=3, sigma=0.5, size=n).clip(1, 120)
num_purchases = np.random.poisson(3, n)

# Create DataFrame
data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Loyalty_Program': loyalty,
    'Annual_Income': income,
    'Prefered_Category': product_category,
    'Average_Price': product_price,
    'Time_Spent': time_spent,
    'Num_Purchases': num_purchases
})

print(data)
data.to_csv('/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/User Recommendataion NN/Dataset/CustomerData.csv', index=True, index_label='ID')