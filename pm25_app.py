import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# App title
st.title("🌫️ PM2.5 Prediction App using AOD & NO₂")

# Instructions
st.markdown("""
Upload two CSV files:
- **PM2.5 Data** (`Date`, `PM2.5`)
- **Satellite Data** (`Date`, `AOD`, `NO2`)
""")

# Upload input files
pm_file = st.file_uploader("Upload PM2.5 Data CSV", type="csv")
sat_file = st.file_uploader("Upload Satellite Data CSV", type="csv")

if pm_file and sat_file:
    # Read files
    df_pm = pd.read_csv(pm_file)
    df_sat = pd.read_csv(sat_file)

    # Merge on Date
    df = pd.merge(df_pm, df_sat, on="Date")
    st.success("Files merged successfully!")

    # Show sample data
    st.subheader("📊 Merged Data Preview")
    st.dataframe(df.head())

    # Prepare data
    X = df[['AOD', 'NO2']]
    y = df['PM2.5']
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Show metrics
    st.subheader("📈 Model Evaluation")
    st.metric(label="RMSE (Root Mean Squared Error)", value=f"{rmse:.2f}")

    # Plot actual vs predicted
    st.subheader("📉 Actual vs Predicted PM2.5")
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual", marker='o')
    ax.plot(y_pred, label="Predicted", marker='x')
    ax.set_xlabel("Sample")
    ax.set_ylabel("PM2.5")
    ax.set_title("Actual vs Predicted PM2.5")
    ax.legend()
    st.pyplot(fig)

    # Feature importance
    st.subheader("🧠 Feature Importance")
    importance = model.feature_importances_
    st.bar_chart(pd.Series(importance, index=X.columns))
else:
    st.info("Please upload both required CSV files to proceed.")