import pandas as pd
import os
from pathlib import Path


data_path = Path('/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/User Recommendataion NN/Dataset/amazon-products-dataset')
product_data = pd.DataFrame()


for _ in data_path.iterdir():
    data = pd.DataFrame(pd.read_csv(_))
    data = data.dropna(subset=['actual_price'])
    sample_data = data.sample(n=int(len(data)/2))
    product_data = pd.concat([product_data, sample_data], ignore_index=True)
    print("Sampling data from:", _)

product_data['discount_price'] = product_data['discount_price'].fillna(0)
product_data['discount_price'] = product_data['discount_price'].replace({'₹':"", ",":""}, regex=True).astype(float)
product_data['actual_price'] = product_data['actual_price'].replace({'₹':"", ",":""}, regex=True).astype(float)
product_data['discount_percentage'] = product_data['discount_price'] / product_data['actual_price']
product_data.drop('Unnamed: 0', axis=1, inplace=True)
print(product_data)
product_data.to_csv('/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/User Recommendataion NN/Dataset/Product_Data.csv', index=True, index_label='Product ID')
