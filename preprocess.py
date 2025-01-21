# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# data = pd.read_csv('dataset/house_prices.csv')

# # Inspect the first few rows of the dataset to understand its structure
# print(data.head())
# print(data.columns)

# # Data preprocessing
# # Update the required columns based on the actual dataset
# required_columns = ['Bedroom AbvGr', 'Full Bath', 'Gr Liv Area', 'Neighborhood']
# missing_columns = [col for col in required_columns if col not in data.columns]

# if missing_columns:
#     raise KeyError(f"Missing columns in the dataset: {missing_columns}")

# data = data.dropna()  # Handle missing values
# X = data[required_columns]
# y = data['SalePrice']

# # Visualize data
# if data.isnull().values.any():
#     sns.heatmap(data.isnull(), cbar=False)
#     plt.show()
# else:
#     print("No missing values in the dataset.")

# print("required_columns:",required_columns)
# # Plot histograms for numerical features with parameters
# X[['Bedroom AbvGr', 'Full Bath', 'Gr Liv Area']].hist(bins=30, figsize=(15, 10), grid=True, alpha=0.7, color='blue')
# plt.show()

# # Plot pairplot for selected features and target variable
# sns.pairplot(data[['Bedroom AbvGr', 'Full Bath', 'Gr Liv Area', 'SalePrice']])
# plt.show()

# # Save preprocessed data
# X.to_csv('X.csv', index=False)
# y.to_csv('y.csv', index=False)

# print("Data preprocessing completed successfully.")


# import pandas as pd

# # Load the dataset
# #file_path = '/mnt/data/AmesHousing.csv'
# data = pd.read_csv('dataset/house_prices.csv')

# # Display the first few rows of the dataset to understand its structure
# data.head(), data.info(), data.describe(include='all')
