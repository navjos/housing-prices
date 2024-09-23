import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import tensorflow_decision_forests as tfdf

df_training = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# this file has 1,460 rows and 81 columns
print(df_training.shape)



# Let's find the correlation of all the features to the target variable
# Correlation calculations require numeric data
# Identify numeric and categorical columns

numeric_columns = df_training.select_dtypes(include=['number']).columns.tolist()
categorical_columns = df_training.select_dtypes(include=['object', 'category']).columns.tolist()

#this encodes the entire dataset to account for categorical variabled
df_encoded = pd.get_dummies(df_training, columns=categorical_columns, drop_first=True)

rows_with_na = df_encoded[df_encoded.isna().any(axis=1)]
# print(rows_with_na)

# Now we realize that theres 339 rows with NA data out of 1460 rows.. 
print('Percent of data with NaN values:', rows_with_na.shape[0]/ df_encoded.shape[0])

# Another more advanced imputation method that uses similiar rows to estimate missing values
imputer = KNNImputer(n_neighbors=5)
df_encoded_imputed = imputer.fit_transform(df_encoded)

# Convert back to DataFrame
df_encoded = pd.DataFrame(df_encoded_imputed, columns=df_encoded.columns)


#compute the correlation matrix
df_encoded_copy1 = df_encoded.copy()
corr_matrix = df_encoded_copy1.corr()

# Get the correlation of each feature with SalePrice
saleprice_corr= corr_matrix['SalePrice'].sort_values(ascending=False)

# Let's now set a correlation level
level = 0.5

# Filter strong correlations and drop 'SalePrice'
strong_correlations = saleprice_corr[abs(saleprice_corr) > level].drop('SalePrice')

print(strong_correlations)

# Next, let's check for multicollinearity among the list of strong_correlation features
selected_features = strong_correlations.index.tolist()

# Create the correlation matrix for selected features
selected_corr_matrix = df_encoded_copy1[selected_features].corr()

#Identify highly correlated (multicollinear) pairs among the selected features that have high correlation
high_corr_pairs = selected_corr_matrix[(abs(selected_corr_matrix) > 0.7) & (selected_corr_matrix!= 1)]
print(high_corr_pairs)

#Identify feautres to drop due to multicollinearity
drop_features = set()

for col in high_corr_pairs.columns:
    for index in high_corr_pairs.index:
        if abs(high_corr_pairs.loc[index, col]) > 0.7 and index != col:
            drop_features.add(col) #add the feature to the collection to drop

# Drop the identified features
df_encoded = df_encoded.drop(columns=drop_features)
print("Dropped features:", drop_features)

# Create a subset of df_encoded with strong correlations and one instance of each multicollinear pair
retained_features = list(set(selected_features) - drop_features)
df_subset_1 = df_encoded_copy1[retained_features]

#Display the final chosen features
print(df_subset_1)

# Alternatively, lets do feature importance from models

df_encoded_copy2 = df_encoded.copy()
corr_matrix_2 = df_encoded_copy2.corr()

# Multicollinearity threshold
threshold = 0.7

# Create a set to hold features to drop
drop_features_2 = set()

# Loop over the correlation matrix and find pairs of highly correlated features
for i in range(len(corr_matrix_2.columns)):
    for j in range(i):
        if abs(corr_matrix_2.iloc[i, j]) > threshold:
            feature_i = corr_matrix_2.columns[i]
            feature_j = corr_matrix_2.columns[j]
            # Add one of the correlated features to the drop list
            drop_features_2.add(feature_j)  # You can choose either feature_j or feature_i


# Drop the highly correlated features from the dataset
df_reduced_2 = df_encoded_copy2.drop(columns=drop_features_2)
print(f"Dropped features due to multicollinearity: {drop_features_2}")

# Step 5: Fit RandomForestRegressor on the reduced dataset
# Define your target variable 'SalePrice' and independent variables
X = df_reduced_2.drop(columns='SalePrice')
y = df_encoded_copy2['SalePrice']

# Initialize and fit the model
model = RandomForestRegressor()
model.fit(X, y)

# Step 6: Extract feature importance
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
# print(feature_importances.sort_values(by='Importance', ascending=False))

# Define the threshold for feature importance
threshold = 0.05

# Filter out features below the threshold
important_features = feature_importances[feature_importances['Importance'] > threshold]

# Get the list of feature names
important_features = important_features['Feature'].tolist()

df_subset_2 = df_encoded_copy2[important_features]

########################## TEST 1 WITH FIRST SET OF FEATURES ###############################
# Assuming df_encoded is your full dataset with features and 'SalePrice' as the target
X = df_subset_1  # Features
y = df_encoded['SalePrice']  # Target

# Split into training and validation sets 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine back into a DataFrame (TF-DF expects a DataFrame)
train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
val_df = pd.concat([X_val, y_val.reset_index(drop=True)], axis=1)

train_df = train_df.dropna(subset=["SalePrice"])
print(train_df)

# # Convert your DataFrame to a TensorFlow dataset for regression
# train_data = tfdf.keras.pd_dataframe_to_tf_dataset(
#     train_df,
#     task=tfdf.keras.Task.REGRESSION,
#     label="SalePrice"
# )

# val_data = tfdf.keras.pd_dataframe_to_tf_dataset(
#     val_df,
#     task=tfdf.keras.Task.REGRESSION,
#     label="SalePrice"
# )

# # Initialize the Random Forest model
# model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION, num_trees=100)

# # Now fit your model
# model.fit(train_data)

# # Optionally evaluate on validation data
# model.evaluate(val_data)
