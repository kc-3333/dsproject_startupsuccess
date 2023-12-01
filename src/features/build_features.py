#----------------------------------------------------------------------------------------------------------
# Import Library
#----------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import MultipleLocator
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#----------------------------------------------------------------------------------------------------------
# Import dataset
#----------------------------------------------------------------------------------------------------------

df = pd.read_csv(r'C:\Users\User\Desktop\Projects\Data Science\project_startup_success\data\processed\df_dp.csv')
pd.set_option('display.max_columns',None) #Display All Columns
df.info()
df.describe()

#----------------------------------------------------------------------------------------------------------
# Feature Engineering
#----------------------------------------------------------------------------------------------------------
df1 = df.copy()
# drop name, market, status, first_funding_at, last_funding_at, founded_at to reduce dimension
df1.drop(columns=['name','market','status','first_funding_at','last_funding_at','founded_at'], inplace=True)
df1.info()

# Additional column for debt_financed
df1['debt_financed'] = df1['debt_financing'].apply(lambda x: 1 if x > 10 else 0)

# 1. Scaling Numerical Features: Apply standardization (z-score scaling) or normalization to scale numerical features, especially for models sensitive to feature scales.
# Extract the relevant numeric columns from your DataFrame
numeric_columns = df1.select_dtypes(include=['float64', 'int64']).drop(columns=['status_score','debt_financed'])
numeric_columns.info()

# Set the figure size for the plots
plt.figure(figsize=(12, 6))

# Create a box plot for each numeric column 
sns.boxplot(data=numeric_columns, orient="h", palette="Set2")
plt.title("Box Plots for Numeric Columns")
plt.xlabel("Value") 

# Distribution (Histogram)
# Create a grid of subplots with proper spacing
fig, axes = plt.subplots(5, 5, figsize=(16, 12))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

# Iterate through the numeric columns and create histograms with normal distribution curves
for i, col in enumerate(numeric_columns):
    row, col_num = divmod(i, 5)
    ax = axes[row, col_num]
    
    # Plot the histogram
    ax.hist(df[col], bins=20, density=True, edgecolor='k', alpha=0.6, color='blue', label='Data')
    
    # Fit a normal distribution to the data
    mu, std = stats.norm.fit(df[col])
    
    # Plot the PDF of the fitted normal distribution
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, color='red', label='Normal Dist.')
    
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    
    ax.legend()

# Hide empty subplots, if any
for i in range(len(numeric_columns), 25):
    row, col_num = divmod(i, 5)
    axes[row, col_num].axis('off')

plt.tight_layout()
plt.show()

# The distribution is skewed. We should not applied z-normalization. But for testing purpose, we still tried.
def z_score_normalization(df, selected_columns):
    for col in selected_columns:
        # Calculate the mean and standard deviation for each column
        mean = df[col].mean()
        std = df[col].std()
        
        # Apply the z-score normalization formula to the column
        df[col] = (df[col] - mean) / std

    return df

df2 = df1.copy()
z_score_normalization(df2,numeric_columns)
df2

# 2. Categorical Variables: Encode categorical variables such as 'market','status','continent,' 'country_code,' 'countinent,' 'region,' and 'industry_group' using one-hot encoding or label encoding based on the data and machine learning algorithm.
# Ordinal encoding,Label encoding, One-Hot Encoding
# One hot encoding as we are dealing with nominal categorical variables, except status
categorical_variables = ['debt_financed','status_score', 'continent','country_code','region','industry_group']
# df1 = pd.get_dummies(df1, columns=categorical_variables, drop_first=True)
# df1.shape #1269 variables, high possibility of overfitting
# try label_encoder
def label_encoder(df,categorical_variables):
    label_encoder = LabelEncoder()
    for col in categorical_variables:
        df[col] = label_encoder.fit_transform(df[col])
    
# Creating dataset
df3 = df1.copy()
df4 = df2.copy()

label_encoder(df3,categorical_variables)
label_encoder(df4,categorical_variables)

# Display the first few rows of the modified DataFrame
df3.head(100)
df3.info()

df4.head(100)
df4.info()

# Datasets for testing
df1 #raw
df2 #z-normalized
df3 #label_encoded
df4 #label_encoded,z-normalized

datasets = [df1, df2, df3, df4]
file_paths = [r'df1_fe.csv', r'df2_fe.csv', r'df3_fe.csv', r'df4_fe.csv']

for dataset, file_path in zip(datasets, file_paths):
    dataset.to_csv(fr'C:\Users\User\Desktop\Projects\Data Science\project_startup_success\data\processed\{file_path}', index=False)