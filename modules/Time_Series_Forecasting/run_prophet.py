# modules/Time_Series_Forecasting/run_prophet_with_optuna.py

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import optuna
import statsmodels.api as sm
import matplotlib.pyplot as plt

def run_prophet(df):
    st.subheader("Prophet Model with Hyperparameter Tuning (Optuna)")

    # --- Initial Settings & Data Preparation ---

    # 日付カラムとターゲットカラムの選択
    st.header("Initial Setting & Quick EDA for Prophet")

    # Select date column
    date_column = st.selectbox("Please select the Date column", df.columns)
    # Select target column
    target_column = st.selectbox("Please select the Target column", [col for col in df.columns if col != date_column])

    # Convert to datetime
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        st.success(f"Successfully converted `{date_column}` to datetime.")
    except Exception as e:
        st.error(f"Error converting `{date_column}` to datetime: {e}")
        st.stop()

    # Set date as index
    df.set_index(date_column, inplace=True)

    # Display initial data overview
    st.subheader("Data Overview")
    st.write(df.head())
    st.write(f"Data points: {len(df)}")

    # Check for missing values
    st.subheader("Missing Values Check")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    if missing_values.sum() > 0:
        st.warning("There are missing values in the dataset. Please handle them before modeling.")
        # Optionally, provide options to handle missing values
        handle_missing = st.selectbox("How would you like to handle missing values?", ["Drop", "Interpolate", "Fill with Mean"])
        if handle_missing == "Drop":
            df = df.dropna()
            st.success("Dropped rows with missing values.")
        elif handle_missing == "Interpolate":
            df = df.interpolate()
            st.success("Interpolated missing values.")
        elif handle_missing == "Fill with Mean":
            df = df.fillna(df.mean())
            st.success("Filled missing values with column means.")

    # Check data types
    st.subheader("Data Types")
    st.write(df.dtypes)

    # Ensure target and exogenous variables are numeric
    st.subheader("Data Type Verification")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

    if non_numeric_cols:
        st.warning("Non-numeric columns detected. They will be excluded from modeling.")

    # Select additional regressors (optional)
    regressor_columns = st.multiselect(
        "Please select Additional Regressors (optional)",
        [col for col in numeric_cols if col != target_column]
    )

    # Set forecast periods
    forecast_periods = st.number_input("Please enter the number of forecast periods (days)", min_value=1, max_value=365, value=30)

    # Select evaluation metric
    evaluation_metric = st.selectbox(
        "Please select the evaluation metric for hyperparameter tuning",
        ["MAE", "MSE", "RMSE"]
    )

    # Run model button
    if st.button("Run Prophet with Optuna Tuning"):
        with st.spinner("Starting hyperparameter tuning with Optuna..."):
            try:
                # Prepare data for Prophet
                prophet_df = df.reset_index()[[date_column, target_column]]
                prophet_df.rename(columns={date_column: 'ds', target_column: 'y'}, inplace=True)

                # Add additional regressors if any
                if regressor_columns:
                    for reg in regressor_columns:
                        prophet_df[reg] = df[reg].values

                # Split data into training and testing sets for evaluation
                train_size = len(prophet_df) - forecast_periods
                if train_size <= 0:
                    st.error("Not enough data points for the specified forecast period. Please reduce the forecast period.")
                    st.stop()

                train = prophet_df.iloc[:train_size]
                test = prophet_df.iloc[train_size:]
                st.write(f"Training data points: {len(train)}")
                st.write(f"Testing data points: {len(test)}")

                # Define the objective function for Optuna
                def objective(trial):
                    # Define hyperparameter search space
                    changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)
                    seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 10.0, log=True)
                    holidays_prior_scale = trial.suggest_float('holidays_prior_scale', 0.01, 10.0, log=True)
                    seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])

                    # Initialize Prophet model with suggested hyperparameters
                    m = Prophet(
                        changepoint_prior_scale=changepoint_prior_scale,
                        seasonality_prior_scale=seasonality_prior_scale,
                        holidays_prior_scale=holidays_prior_scale,
                        seasonality_mode=seasonality_mode
                    )

                    # Add additional regressors to the model
                    if regressor_columns:
                        for reg in regressor_columns:
                            m.add_regressor(reg)

                    # Fit the model
                    m.fit(train)

                    # Create future dataframe
                    future = m.make_future_dataframe(periods=forecast_periods)

                    # If additional regressors exist, add them to the future dataframe
                    if regressor_columns:
                        # Ensure that the forecast_periods do not exceed the available exog data
                        if len(df) < forecast_periods:
                            return float('inf')  # Invalid configuration
                        future_exog = df[regressor_columns].iloc[-forecast_periods:].copy()
                        future[regressor_columns] = future_exog.values

                    # Predict
                    forecast = m.predict(future)

                    # Extract forecast for the test period
                    forecast_test = forecast.iloc[-forecast_periods:][['ds', 'yhat']]

                    # Merge with actual test data
                    comparison_df = test.merge(forecast_test, on='ds')

                    # Calculate the selected evaluation metric
                    if evaluation_metric == "MAE":
                        metric_value = mean_absolute_error(comparison_df['y'], comparison_df['yhat'])
                    elif evaluation_metric == "MSE":
                        metric_value = mean_squared_error(comparison_df['y'], comparison_df['yhat'])
                    elif evaluation_metric == "RMSE":
                        metric_value = np.sqrt(mean_squared_error(comparison_df['y'], comparison_df['yhat']))
                    else:
                        metric_value = mean_absolute_error(comparison_df['y'], comparison_df['yhat'])  # Default to MAE

                    return metric_value

                # Create Optuna study and optimize
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=50)

                # Get the best hyperparameters
                best_params = study.best_params
                st.success("Hyperparameter tuning completed!")

                st.write("### Best Hyperparameters Found")
                st.write(best_params)
                st.write(f"Best {evaluation_metric}: {study.best_value:.4f}")

                # Train the final model with the best hyperparameters
                m = Prophet(
                    changepoint_prior_scale=best_params['changepoint_prior_scale'],
                    seasonality_prior_scale=best_params['seasonality_prior_scale'],
                    holidays_prior_scale=best_params['holidays_prior_scale'],
                    seasonality_mode=best_params['seasonality_mode']
                )

                # Add additional regressors to the model
                if regressor_columns:
                    for reg in regressor_columns:
                        m.add_regressor(reg)

                # Fit the final model
                m.fit(train)

                # Create future dataframe
                future = m.make_future_dataframe(periods=forecast_periods)

                # If additional regressors exist, add them to the future dataframe
                if regressor_columns:
                    # Extract exogenous variables for the forecast period
                    future_exog = df[regressor_columns].iloc[-forecast_periods:].copy()
                    future[regressor_columns] = future_exog.values

                # Predict
                forecast = m.predict(future)

                # Extract forecast for the test period
                forecast_test = forecast.iloc[-forecast_periods:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

                # Merge with actual test data
                comparison_df = test.merge(forecast_test, on='ds')

                # Plot forecast vs actuals using Plotly
                fig_forecast = go.Figure()

                # Observed data (Training data)
                fig_forecast.add_trace(go.Scatter(
                    x=train['ds'],
                    y=train['y'],
                    mode='lines',
                    name='Training Data',
                    line=dict(color='grey')
                ))

                # Observed data (Testing data)
                fig_forecast.add_trace(go.Scatter(
                    x=test['ds'],
                    y=test['y'],
                    mode='lines',
                    name='Actual Test Data',
                    line=dict(color='grey')
                ))

                # Forecasted data (Predictions)
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_test['ds'],
                    y=forecast_test['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='orangered')
                ))

                # 信頼区間の描画
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_test['ds'],
                    y=forecast_test['yhat_upper'],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Upper Confidence Interval',
                    showlegend=False
                ))

                fig_forecast.add_trace(go.Scatter(
                    x=forecast_test['ds'],
                    y=forecast_test['yhat_lower'],
                    mode='lines',
                    fill='tonexty',  # 上限と下限の間を塗りつぶす
                    fillcolor='rgba(255, 69, 0, 0.2)',  # 透明なオレンジ色
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Lower Confidence Interval',
                    showlegend=False
                ))

                # Update layout for better interactivity
                fig_forecast.update_layout(
                    title="Forecast Results vs Actual",
                    xaxis_title="Date",
                    yaxis_title=target_column,
                    hovermode="x unified"
                )

                # Add range slider and buttons for range selection
                fig_forecast.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )

                # Display the Plotly forecast plot
                st.plotly_chart(fig_forecast, use_container_width=True)

                # Evaluate forecast accuracy using the test set
                st.subheader("Forecast Accuracy Metrics")
                if len(test) > 0:
                    y_true = comparison_df['y']
                    y_pred = comparison_df['yhat']
                    mse = mean_squared_error(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                    # Create a DataFrame for metrics
                    metrics = {
                        "Metric": ["MSE", "MAE", "RMSE", "MAPE"],
                        "Value": [mse, mae, rmse, mape]
                    }
                    df_metrics = pd.DataFrame(metrics)

                    # Display metrics as a table
                    st.table(df_metrics)
                else:
                    st.warning("Insufficient data for evaluating forecast accuracy. Please reduce the forecast period.")


            except Exception as e:
                st.error(f"Error during Prophet modeling with Optuna: {e}")
                st.write("Please refer to the Prophet documentation: [Prophet Documentation](https://facebook.github.io/prophet/docs/quick_start.html)")
