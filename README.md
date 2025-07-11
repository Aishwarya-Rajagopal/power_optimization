⚡ Power Consumption Forecasting – Wellington, NZ 🌤️

This machine learning project predicts **Zone A power consumption** in Wellington, New Zealand, using meteorological and environmental data. The goal is to assist energy resource planning and demand optimization.

📈 Project Objective

> To build a robust machine learning pipeline that  forecasts power consumption using key environmental and meteorological inputs such as temperature, humidity, wind speed, solar radiation, and air quality.

🛠️ Tools & Technologies

- **Languages & Libraries**: Python, Pandas, NumPy, Scikit-learn, XGBoost, ExtraTrees, Joblib
- **Model Deployment**: Streamlit (Web App), Streamlit Cloud 
- **EDA & Visualization**: Seaborn, Matplotlib
- **Environments**: Conda, VSCode , Google Colab
- **Version Control**: GitHub

🧠 Skills & Best Practices Demonstrated

✅ **Data Cleaning & Feature Engineering**  
  - Interaction terms, log and polynomial transformations
  - Outlier handling using RobustScaler
  - Missing Value imputation
    
✅ **Exploratory Data Analysis (EDA)**  
  - Boxplots, correlation heatmaps, skewness correction
  - Categorization and encoding of variable columns
    
✅ **ML Modeling & Evaluation**  
  - Compared 6+ models including Linear regression, Random Forest, XGBoost, ExtraTrees  
  - Final model: **Stacking Regressor** with linear regression as meta-model  
  - train-test split before scaling to prevent data leakage
  - Overfitting detection with cross validation scores
    
✅ **Model Optimization**  
  - GridSearchCV, feature importance analysis
    
✅ **Deployment**  
  - Web app built using **Streamlit**  
  - Final model deployed online with clean UI and user inputs

✅ Final Results

 Model : StackingRegressor(ExtraTrees + XGBoost + LinearRegression)

| Metric | Value |
|--------|--------|
| Test R² | **0.6431** |
| RMSE    | **4792.61** |
| MAE     | **3291.09** |

