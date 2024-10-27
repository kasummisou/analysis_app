import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import holidays

# Install if necessary
# !pip install holidays

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def mase(y_true, y_pred, naive_forecast=None):
    """Mean Absolute Scaled Error"""
    n = len(y_true)
    if naive_forecast is None:
        d = np.abs(np.diff(y_true)).sum() / (n - 1)
    else:
        d = np.mean(np.abs(y_true - naive_forecast))
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

def create_features(df, date_col, target_col):
    st.subheader("Feature Engineering")

    # Create Domain-Specific Features
    st.markdown("### Domain-Specific Features")
    # Example: Flags for specific events or promotion periods
    # Allow users to input event dates
    event_dates = st.text_area("Please enter event dates separated by commas (e.g., 2023-12-25,2024-01-01)")
    if event_dates:
        try:
            event_dates = [pd.to_datetime(date.strip()) for date in event_dates.split(",")]
            df['Event_Flag'] = df.index.isin(event_dates).astype(int)
            st.success("Domain-Specific Feature (Event_Flag) has been created.")
        except Exception as e:
            st.error(f"Error processing event dates: {e}")

    # Create Calendar Features
    st.markdown("### Calendar Features")
    # Weekday, Month, Quarter, Holiday flags, etc.
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    # Holiday flag (for Japan)
    jp_holidays = holidays.JP()
    df['Holiday'] = df.index.isin(jp_holidays).astype(int)
    st.success("Calendar Features have been created.")

    # Create Lag Features
    st.markdown("### Lag Features")
    lag_steps = st.number_input("Enter the number of lag steps (e.g., 1,2,3)", min_value=1, max_value=30, value=1, step=1)
    df[f'Lag_{lag_steps}'] = df[target_col].shift(lag_steps)
    st.success(f"Lag Feature (Lag_{lag_steps}) has been created.")

    # Create Rolling Features
    st.markdown("### Rolling Features")
    rolling_window = st.number_input("Enter the rolling window size (e.g., 3,7,14)", min_value=1, max_value=365, value=7, step=1)
    df[f'Rolling_Mean_{rolling_window}'] = df[target_col].rolling(window=rolling_window).mean()
    df[f'Rolling_STD_{rolling_window}'] = df[target_col].rolling(window=rolling_window).std()
    st.success(f"Rolling Features (Rolling_Mean_{rolling_window}, Rolling_STD_{rolling_window}) have been created.")

    # Create Expanding Features
    st.markdown("### Expanding Features")
    df['Expanding_Mean'] = df[target_col].expanding().mean()
    df['Expanding_STD'] = df[target_col].expanding().std()
    st.success("Expanding Features (Expanding_Mean, Expanding_STD) have been created.")

    return df

def visualize_features(df, target_col, features):
    st.subheader("Feature Visualization")

    # Time Series Plot
    st.markdown("### Time Series Plot")
    selected_features = st.multiselect("Select features to visualize", features, default=features[:3])
    if selected_features:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df.index, df[target_col], label=target_col, color='blue')
        for feature in selected_features:
            ax.plot(df.index, df[feature], label=feature)
        ax.set_title("Time Series and Features")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

    # Correlation Matrix
    st.markdown("### Correlation Matrix")
    if len(features) > 0:
        corr = df[features].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)
    else:
        st.write("No features selected for displaying the correlation matrix.")

    # Heatmap
    st.markdown("### Heatmap")
    for feature in selected_features:
        fig_heat, ax_heat = plt.subplots(figsize=(14, 2))
        sns.heatmap(df[[feature]].T, cmap='viridis', cbar=True, ax=ax_heat)
        ax_heat.set_title(f"Heatmap of {feature}")
        st.pyplot(fig_heat)

def visualization_of_time_series_feature(df):
    st.title("Time Series Feature Engineering and Visualization App")

    st.markdown("""
    This app allows you to create and visualize the following features from your uploaded time series data:
    - Domain-Specific Features
    - Calendar Features
    - Lag Features
    - Rolling Features
    - Expanding Features
    """)

    try:
        st.success("Data successfully loaded and passed to the feature engineering function.")
        st.write("Data Preview:")
        st.write(df.head())

        # Select Date and Target Columns
        st.subheader("Settings")
        date_column = st.selectbox("Please select the Date column", df.columns)
        target_column = st.selectbox("Please select the Target column", [col for col in df.columns if col != date_column])

        # Convert Date Column to datetime if not already
        if not np.issubdtype(df[date_column].dtype, np.datetime64):
            try:
                df[date_column] = pd.to_datetime(df[date_column])
                st.success(f"Converted `{date_column}` to datetime.")
            except Exception as e:
                st.error(f"Error converting `{date_column}` to datetime: {e}")
                return

        # Set Date as Index
        df.set_index(date_column, inplace=True)
        st.write("Set Date column as index.")
        st.write(df.head())

        # Feature Engineering
        df = create_features(df, date_column, target_column)

        # Visualization
        feature_columns = df.columns.tolist()
        if target_column in feature_columns:
            feature_columns.remove(target_column)
        visualize_features(df, target_column, feature_columns)

        # Download Data with Features (Optional)
        st.subheader("Download Data with Features")
        csv = df.to_csv().encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='feature_engineered_data.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"An error occurred during feature engineering and visualization: {e}")
