# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:28:31 2024
@author: Palvi
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

############################# Load and Clean Datasets ##########################################

# Load datasets
metadata_path = 'na_metadata.csv'
weather_path = 'na_weather.csv'
electricity_path = 'updated_transposed_daily_electricity_usage.csv'

metadata_df = pd.read_csv(metadata_path)
weather_df = pd.read_csv(weather_path)
electricity_df = pd.read_csv(electricity_path)

# Drop unnecessary columns from metadata
columns_to_drop_metadata = [
    'building_id_kaggle', 'site_id_kaggle', 'solar', 'industry', 'subindustry',
    'heatingtype', 'date_opened', 'numberoffloors', 'occupants', 'energystarscore',
    'eui', 'site_eui', 'source_eui', 'leed_level', 'rating'
]
metadata_df_cleaned = metadata_df.drop(columns=columns_to_drop_metadata, errors='ignore')

# Convert weather timestamp to datetime
weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], format='%d-%m-%Y %H:%M')

# Reshape electricity usage dataset into long format
electricity_long = electricity_df.melt(
    id_vars=['building_name', 'site_id'],
    var_name='date',
    value_name='electricity_usage'
)
electricity_long['date'] = pd.to_datetime(electricity_long['date'], format='%d-%m-%Y')

# Merge electricity usage with metadata and weather data
electricity_metadata_merged = electricity_long.merge(
    metadata_df_cleaned,
    left_on=['building_name', 'site_id'],
    right_on=['building_id', 'site_id'],
    how='left'
)
final_merged_dataset = electricity_metadata_merged.merge(
    weather_df,
    left_on=['site_id', 'date'],
    right_on=['site_id', 'timestamp'],
    how='left'
)
final_merged_dataset.drop(columns=['timestamp'], inplace=True)

# Drop irrelevant columns
columns_to_drop_final = [
    'hotwater', 'chilledwater', 'steam', 'water', 'irrigation', 'gas', 'unique_space_usages'
]
final_merged_dataset_cleaned = final_merged_dataset.drop(columns=columns_to_drop_final, errors='ignore')

####################### Feature Engineering ####################################################

# Add time-based features
final_merged_dataset_cleaned['month'] = final_merged_dataset_cleaned['date'].dt.month
final_merged_dataset_cleaned['day_of_week'] = final_merged_dataset_cleaned['date'].dt.dayofweek
final_merged_dataset_cleaned['is_weekend'] = final_merged_dataset_cleaned['day_of_week'].isin([5, 6]).astype(int)

# Add lag features
final_merged_dataset_cleaned['lag_1'] = final_merged_dataset_cleaned['electricity_usage'].shift(1)
final_merged_dataset_cleaned['lag_7'] = final_merged_dataset_cleaned['electricity_usage'].shift(7)
final_merged_dataset_cleaned.dropna(subset=['lag_1', 'lag_7'], inplace=True)


################## Visualizing Building Electricity Usage Trends ##################################################

# Aggregate electricity usage over time (daily trends across all buildings)
aggregated_data = final_merged_dataset_cleaned.groupby('date')['electricity_usage'].sum().reset_index()

# Plot the aggregated electricity usage trend
plt.figure(figsize=(14, 8))
plt.plot(aggregated_data['date'], aggregated_data['electricity_usage'], color='purple', linewidth=2, alpha=0.8)
plt.title('Aggregated Daily Electricity Usage Trend', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Electricity Usage (kWh)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Select a sample of buildings to visualize electricity usage trends
sample_buildings = final_merged_dataset_cleaned['building_name'].dropna().unique()[:5]  # Adjust the number as needed

# Filter data for these buildings
sample_data = final_merged_dataset_cleaned[final_merged_dataset_cleaned['building_name'].isin(sample_buildings)]

# Plot electricity usage trends for each building
plt.figure(figsize=(14, 8))
for building in sample_buildings:
    building_data = sample_data[sample_data['building_name'] == building]
    plt.plot(
        building_data['date'], 
        building_data['electricity_usage'], 
        label=building
    )

plt.title('Electricity Usage Trends for Sample Buildings', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Electricity Usage (kWh)', fontsize=12)
plt.legend(title="Building Name", fontsize=10)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

############################### Outlier Removal #######################################################

# Detect and remove outliers using IQR
Q1 = final_merged_dataset_cleaned['electricity_usage'].quantile(0.25)
Q3 = final_merged_dataset_cleaned['electricity_usage'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

final_merged_dataset_cleaned = final_merged_dataset_cleaned[
    (final_merged_dataset_cleaned['electricity_usage'] >= lower_bound) &
    (final_merged_dataset_cleaned['electricity_usage'] <= upper_bound)
]

########################## Handle Missing Values #################################################

# Impute missing values for predictors
predictors = [
    'month', 'day_of_week', 'is_weekend', 'airTemperature',
    'dewTemperature', 'seaLvlPressure', 'windSpeed', 'sqm', 'lag_1', 'lag_7'
]

final_merged_dataset_cleaned[predictors].isnull().sum()

# Fill missing values with 0 for the predictors
final_merged_dataset_cleaned[predictors] = final_merged_dataset_cleaned[predictors].fillna(0)

############################# Random Forest Regression ##############################################

# Train-test split
X = final_merged_dataset_cleaned[predictors]
y = final_merged_dataset_cleaned['electricity_usage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) #Time Taken: Approx 10 mins

# Predict and evaluate
y_pred = rf_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")
print(f"Mean Absolute Error: {mae}")
# Output 
# Root Mean Squared Error: 386.3929068829207
# R^2 Score: 0.9610776214968123
# Mean Absolute Error: 167.09625589991595


# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Importance - Random Forest', fontsize=16)
plt.tight_layout()
plt.show()


########################### Cross Validation #################################################

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Perform k-fold cross-validation on the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Use negative mean squared error for scoring
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
# Time Taken: Between 30-45 mins 

# Convert scores to positive RMSE
rmse_scores = np.sqrt(-cv_scores)

print("Cross-Validation Results:")
print(f"RMSE for each fold: {rmse_scores}")
print(f"Mean RMSE: {rmse_scores.mean()}")
print(f"Standard Deviation of RMSE: {rmse_scores.std()}")

# OUTPUT
# Cross-Validation Results:
# RMSE for each fold: [806.21498444 607.50697149 620.53994971 590.9044764  879.38227603]
# Mean RMSE: 700.9097316140258
# Standard Deviation of RMSE: 118.51259763968248

# Evaluate overall model performance using cross-validation
mean_rmse = rmse_scores.mean()
std_rmse = rmse_scores.std()

print(f"Cross-Validated RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")
# OUTPUT
# Cross-Validated RMSE: 700.91 ± 118.51


############################# Visualization ##############################################3

# Plot a subset of the data
size = 500  # Adjust the number of points to display
# Apply rolling mean for smoothing
rolling_window = 50
y_test_rolling = y_test.reset_index(drop=True).rolling(window=rolling_window).mean()
y_pred_rolling = pd.Series(y_pred).rolling(window=rolling_window).mean()

plt.figure(figsize=(16, 12))

# Subplot 1: Actual vs Predicted (Raw)
plt.subplot(2, 1, 1)
plt.plot(y_test.reset_index(drop=True)[:size], label='Actual (Raw)', color='blue', linewidth=1.5, alpha=0.8)
plt.plot(y_pred[:size], label='Predicted (Raw)', color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
plt.title(f'Actual vs Predicted Electricity Usage (First {size} Samples)', fontsize=16)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Electricity Usage (kWh)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Smoothed Actual vs Predicted
plt.subplot(2, 1, 2)
plt.plot(y_test_rolling[:size], label='Actual (Smoothed)', color='blue', linewidth=2, alpha=0.9)
plt.plot(y_pred_rolling[:size], label='Predicted (Smoothed)', color='orange', linestyle='--', linewidth=2, alpha=0.9)
plt.title('Smoothed Actual vs Predicted Electricity Usage', fontsize=16)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Electricity Usage (kWh)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)  # Reference line
plt.title('Predicted vs Actual Electricity Usage', fontsize=16, fontweight='bold')
plt.xlabel('Actual Electricity Usage (kWh)', fontsize=12)
plt.ylabel('Predicted Electricity Usage (kWh)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()







