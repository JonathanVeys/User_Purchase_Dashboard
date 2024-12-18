import pandas as pd
import numpy as np
from scipy.stats import lognorm

user_df = pd.read_csv('/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/User Recommendataion NN/Dataset/CustomerData.csv')
product_df = pd.read_csv('/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/User Recommendataion NN/Dataset/Product_Data.csv')

# Assuming user_df and product_df exist

# Generate purchase data
def generate_purchases(user_df, product_df, num_purchases):
    purchase_data = []
    idx = 1
    
    while idx <= num_purchases:
        # Randomly select a user
        user = user_df.sample(1).iloc[0]
        
        # Randomly select a product
        product = product_df.sample(1).iloc[0]
        
        # Calculate purchase probability based on user and product attributes
        base_prob = 0.1  # Base probability of purchase
        if user['Loyalty_Program']:
            base_prob += 0.15
        base_prob += min(user['Annual_Income'] / 100_000, 0.1)  # Income factor
        if user['Prefered_Category'] == product['main_category']:
            base_prob += 0.15
        if product['discount_percentage']:
            base_prob += 0.15*product['discount_percentage']
        age_prob = lognorm.pdf(user['Age'], 0.5, scale=np.exp(3.4))
        base_prob += 5*age_prob

        prob = 2*np.random.rand()
        print("User: ",idx," Purchase probability: ", prob, " User Prob: ", base_prob)
        idx += 1
        # Make a purchase decision
        if prob < base_prob:
            purchase_data.append({
                'User ID': user['ID'],
                'Product ID': product['Product ID'],
                'Age': user['Age'],
                'Loyalty_Program': user['Loyalty_Program'],
                'Annual_Income': user['Annual_Income'],
                'Prefered_Category': user['Prefered_Category'],
                'Product Category': product['main_category'],
                'Price Paid': product['discount_price'] if product['discount_price'] else product['actual_price'],
                'discount_percentage': product['discount_percentage'],
                'purchase': 1
            })
        else:
            purchase_data.append({
                'User ID': user['ID'],
                'Product ID': product['Product ID'],
                'Age': user['Age'],
                'Loyalty_Program': user['Loyalty_Program'],
                'Annual_Income': user['Annual_Income'],
                'Prefered_Category': user['Prefered_Category'],
                'Product Category': product['main_category'],
                'Price Paid': product['discount_price'] if product['discount_price'] else product['actual_price'],
                'discount_percentage': product['discount_percentage'],
                'purchase': 0
            })
    purchase_data = pd.DataFrame(purchase_data)
    purchase_data['Prefered_Category'] = np.where(purchase_data['Prefered_Category'] == purchase_data['Product Category'], 1, 0)
    purchase_data.drop('Product Category', axis=1, inplace=True)
    return pd.DataFrame(purchase_data)

# Example usage
num_purchases = 100000
purchase_df = generate_purchases(user_df, product_df, num_purchases)

print(purchase_df)
successful_purchases = purchase_df['purchase'].sum()
print(successful_purchases)
purchase_df.to_csv('/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/User Recommendataion NN/Dataset/purchase_dataset.csv')