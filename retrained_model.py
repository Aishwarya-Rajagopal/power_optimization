import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import joblib

# Load your dataset
data = pd.read_excel("/content/City Power Consumption.xlsx",index_col='S no')  

# Temperature colm - fixing typos
data['Temperature'] = np.where(data['Temperature']=='5.488 dc',5.488,data['Temperature'])
data['Temperature'] = np.where(data['Temperature']=='13.65 CD',13.65,data['Temperature'])
data['Temperature'] = np.where(data['Temperature']=='12.31 dc',12.31,data['Temperature'])

# Temperature colm - fixing typos
data['Humidity'] = np.where(data['Humidity']=='84.8 i',84.8,data['Humidity'])



# converting to numeric dtype
data['Temperature'] = data['Temperature'].astype(float)
data['Humidity'] = data['Humidity'].astype(float)

data = data.rename(columns = {' Power Consumption in Zone A':'Power Consumption in Zone A'})

from sklearn.impute import SimpleImputer


features_to_impute = data.drop(columns=['Cloudiness', 'Power Consumption in Zone A'])


imputer = SimpleImputer(strategy='median')


imputed_data = imputer.fit_transform(features_to_impute)

# Convert back to DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=features_to_impute.columns)

# Replace original columns with imputed ones
data.update(imputed_df)

data['Temperature'] = np.where(data['Temperature'].isnull(),data['Temperature'].median(),data['Temperature'])


def categorize_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

data['AQI_Category'] = data['Air Quality Index (PM)'].apply(categorize_aqi)

# ordinal encoding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['AQI_Category_Encoded'] = le.fit_transform(data['AQI_Category']).astype(int)

data = data.drop(['AQI_Category'],axis=1)

data['general diffuse flows'] = np.log1p(data['general diffuse flows'])
data['diffuse flows'] = np.log1p(data['diffuse flows'])
data['Wind Speed'] = np.sqrt(data['Wind Speed'])


# Separate features and target
from sklearn.model_selection import train_test_split

x = data.drop('Power Consumption in Zone A', axis=1)
y = data['Power Consumption in Zone A']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=5)

# handling outliers

from sklearn.preprocessing import RobustScaler

# Columns to scale
features_to_scale = ['Temperature', 'Humidity', 'general diffuse flows',
                     'diffuse flows', 'Wind Speed']


r_scaler = RobustScaler()

# copying train_test data for scaling
x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()

x_train_scaled[features_to_scale] = r_scaler.fit_transform(x_train[features_to_scale])
x_test_scaled[features_to_scale] = r_scaler.transform(x_test[features_to_scale])


# Optional: re-apply feature engineering if needed
# Create new features
x_train_fe = x_train_scaled.copy()
x_test_fe = x_test_scaled.copy()

# Add Temperature + Humidity
x_train_fe['Temp_Humidity_Interaction'] = x_train_fe['Temperature'] * x_train_fe['Humidity']
x_test_fe['Temp_Humidity_Interaction'] = x_test_fe['Temperature'] * x_test_fe['Humidity']

# Temperature squared
x_train_fe['Temperature_Sq'] = x_train_fe['Temperature'] ** 2
x_test_fe['Temperature_Sq'] = x_test_fe['Temperature'] ** 2



# Define stacking model
base_models = [
    ('et', ExtraTreesRegressor(random_state=42)),
    ('xgb', XGBRegressor(random_state=42, verbosity=0))
]

meta_model = LinearRegression()

# Build Stacking Regressor
stack_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=10,
    n_jobs=-1
)

stack_model.fit(x_train_fe, y_train)

# Save model (this will now match your current environment)
joblib.dump(stack_model, "power_consumption_model.pkl")

print("✅ Model retrained and saved successfully!")
