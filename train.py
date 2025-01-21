# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load preprocessed data
# X = pd.read_csv('X.csv')
# y = pd.read_csv('y.csv')

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# mae = mean_absolute_error(y_test, y_pred)
# print(f'Mean Absolute Error: {mae}')

# # Visualize correlation matrix
# corr_matrix = X.corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.show()

# # Visualize feature distributions
# sns.boxplot(x='bedrooms', y='price', data=pd.concat([X, y], axis=1))
# plt.show()

# # Save the model
# import joblib
# joblib.dump(model, 'model.pkl')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import pandas as pd

# Load the dataset
#file_path = '/mnt/data/AmesHousing.csv'
data = pd.read_csv('dataset/house_prices.csv')

# Display the first few rows of the dataset to understand its structure
data.head(), data.info(), data.describe(include='all')


# Step 1: Handle missing values
# Fill numerical missing values with the median
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

# Fill categorical missing values with the mode
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Step 2: Encode categorical variables
encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Step 3: Normalize numerical features
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Step 4: Split the data into train and test sets
X = data.drop(columns=['SalePrice', 'Order', 'PID'])  # Exclude target and non-relevant columns
y = data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check processed data shapes
X_train.shape, X_test.shape, y_train.shape, y_test.shape
