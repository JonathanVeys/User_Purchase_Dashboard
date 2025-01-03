import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/User Recommendataion NN/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.drop('customerID', axis='columns', inplace=True)
data = data[data.TotalCharges!=" "]
data.TotalCharges = pd.to_numeric(data.TotalCharges)
tenureChurnNo = data[data.Churn == 'No'].MonthlyCharges
tenureChurnYes = data[data.Churn == 'Yes'].MonthlyCharges
plt.hist([tenureChurnYes, tenureChurnNo], color=['green', 'red'], label=['Churn=Yes', 'Churn=No'])
plt.xlabel('Monthy Charge')
plt.ylabel('Number of customers')
plt.legend()
# plt.show()

def print_unique_col_values(data):
    for col in data:
        # if data[col].dtypes=='object':
            print(f'{col}: {data[col].unique()}')
data.replace('No internet service', 'no', inplace=True)
data.replace('No phone service', 'No', inplace=True)
data.replace('no', 'No', inplace=True)
data['gender'].replace({'Female':1, 'Male':0}, inplace=True)

data = pd.get_dummies(data=data, columns=['InternetService', 'Contract', 'PaymentMethod'])


yes_no_columns = ['Partner', 'Dependants', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
for column in yes_no_columns:
    data.replace({'Yes':1, 'No':0}, inplace=True)

bool_columns = ['InternetService_DSL', 'InternetService_FiberOptic', 'InternetService_No', 'Contranct_Month_to_month', 'Contract_One year', 'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
for column in bool_columns:
    data.replace({True:1, False:0}, inplace=True)

columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = MinMaxScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

from sklearn.utils import resample
import pandas as pd

# Separate the majority and minority classes
churn_yes = data[data['Churn'] == 1]
churn_no = data[data['Churn'] == 0]

# Oversample the minority class
churn_yes_oversampled = resample(churn_yes,
                                 replace=True,  # Sample with replacement
                                 n_samples=len(churn_no),  # Match number of majority samples
                                 random_state=42)  # Reproducibility

# Combine majority class with oversampled minority class
balanced_data = pd.concat([churn_no, churn_yes_oversampled])

# Shuffle the dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)


balanced_data.to_csv('/Users/jonathancrocker/Documents/VSCode/Code/Pytorch/User Recommendataion NN/Dataset/Processesed-Churn-Data.csv', index=False)
