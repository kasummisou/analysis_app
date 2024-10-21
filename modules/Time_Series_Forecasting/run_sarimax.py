import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go  # Plotlyのインポート
import plotly.express as px  # Plotly Expressのインポート

def run_sarimax(df):
    st.subheader("SARIMAX Model")
    
    # Selection of date and target columns
    st.header("Initial Setting & Quick EDA for SARIMAX")
    
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
    
    # Select ID columns (optional)
    id_column = st.multiselect("Please select ID columns (optional)", df.columns.tolist())
    
    # Select exogenous variables (optional)
    exogenous_columns = st.multiselect(
        "Please select Exogenous variables (optional)",
        [col for col in numeric_cols if col not in id_column + [target_column]]
    )
    
    # Set forecast periods
    forecast_periods = st.number_input("Please enter the number of forecast periods (days)", min_value=1, max_value=365, value=30)
    
    # Select parameter setting method
    param_method = st.radio("Please select the parameter setting method", ["Manual", "AutoARIMA"])
    
    # Define seasonal_period outside the if-else to ensure it's always defined
    seasonal_period = st.number_input("Seasonal Period (m)", min_value=1, max_value=24, value=12)
    
    if param_method == "Manual":
        st.write("Please set SARIMAX parameters:")
        p = st.number_input("Order of AR (p)", min_value=0, max_value=5, value=1)
        d = st.number_input("Order of Integration (d)", min_value=0, max_value=2, value=1)
        q = st.number_input("Order of MA (q)", min_value=0, max_value=5, value=1)
        seasonal_p = st.number_input("Seasonal AR (P)", min_value=0, max_value=5, value=1)
        seasonal_d = st.number_input("Seasonal Integration (D)", min_value=0, max_value=2, value=1)
        seasonal_q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=5, value=1)
    else:
        st.write("Using AutoARIMA to automatically determine SARIMAX parameters.")
    
    # Run model button
    if st.button("Run SARIMAX"):
        with st.spinner("Fitting SARIMAX model..."):
            try:
                # Prepare data
                selected_columns = [target_column] + exogenous_columns
                df_filtered = df[selected_columns].copy()
                
                # Handle exogenous variables if any
                if exogenous_columns:
                    exog = df_filtered[exogenous_columns]
                else:
                    exog = None
                
                # Check if target variable is constant
                if df_filtered[target_column].nunique() == 1:
                    st.error("The target variable is constant. ARIMA models require variability in the target.")
                    st.stop()
                
                # Check stationarity using Augmented Dickey-Fuller test
                st.subheader("Stationarity Check (ADF Test)")
                adf_result = adfuller(df_filtered[target_column].dropna())
                st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                st.write(f"p-value: {adf_result[1]:.4f}")
                if adf_result[1] < 0.05:
                    st.write("The series is stationary.")
                else:
                    st.write("The series is non-stationary. Differencing will be applied if needed.")
                
                # Split data into training and testing sets for evaluation
                st.subheader("Train-Test Split")
                train_size = len(df_filtered) - forecast_periods
                if train_size <= 0:
                    st.error("Not enough data points for the specified forecast period. Please reduce the forecast period.")
                    st.stop()
                train = df_filtered.iloc[:train_size]
                test = df_filtered.iloc[train_size:]
                st.write(f"Training data points: {len(train)}")
                st.write(f"Testing data points: {len(test)}")
                
                # Handle exogenous variables for training and testing
                if exogenous_columns:
                    exog_train = train[exogenous_columns]
                    exog_test = test[exogenous_columns]
                else:
                    exog_train = None
                    exog_test = None
                
                # Fit model
                if param_method == "Manual":
                    model = SARIMAX(
                        train[target_column],
                        order=(p, d, q),
                        seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_period),
                        exog=exog_train,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                else:
                    # Use AutoARIMA to find the best SARIMAX parameters based on default scoring (AIC)
                    auto_model = auto_arima(
                        train[target_column],
                        exogenous=exog_train,
                        seasonal=True,
                        m=seasonal_period,
                        stepwise=True,
                        suppress_warnings=True,
                        trace=False,
                        error_action='ignore',
                        max_p=5,
                        max_q=5,
                        max_P=5,
                        max_Q=5,
                        max_order=10,
                        stationary=False,
                        information_criterion='aic',
                        n_jobs=-1
                    )
                    model = SARIMAX(
                        train[target_column],
                        order=auto_model.order,
                        seasonal_order=auto_model.seasonal_order,
                        exog=exog_train,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    st.write(f"Best parameters found by AutoARIMA: order={auto_model.order}, seasonal_order={auto_model.seasonal_order}")
                
                result = model.fit(disp=False)
                
                # Execute forecast on the test set
                pred = result.get_forecast(steps=forecast_periods, exog=exog_test)
                pred_ci = pred.conf_int()
                pred_mean = pred.predicted_mean
                
                # Plot forecast results using Plotly
                fig_forecast = go.Figure()
                
                # Observed data (Training data)
                fig_forecast.add_trace(go.Scatter(
                    x=train.index,
                    y=train[target_column],
                    mode='lines',
                    name='Training Data',
                    line=dict(color='grey')
                ))
                
                # Observed data (Testing data)
                fig_forecast.add_trace(go.Scatter(
                    x=test.index,
                    y=test[target_column],
                    mode='lines',
                    name='Actual Test Data',
                    line=dict(color='grey')
                ))
                
                # Forecasted data (Predictions)
                fig_forecast.add_trace(go.Scatter(
                    x=pred_mean.index,
                    y=pred_mean,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='orangered')
                ))
                
                # 信頼区間の描画方法を修正
                # 上限をプロット
                fig_forecast.add_trace(go.Scatter(
                    x=pred_ci.index,
                    y=pred_ci.iloc[:, 1],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Upper Confidence Interval',
                    showlegend=False
                ))
                
                # 下限をプロットし、上限と下限の間を塗りつぶす
                fig_forecast.add_trace(go.Scatter(
                    x=pred_ci.index,
                    y=pred_ci.iloc[:, 0],
                    # mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255, 69, 0, 0.2)',  # Transparent orangered
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
                    y_true = test[target_column]
                    y_pred = pred_mean
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
                
                # Display model summary in an expandable section
                with st.expander("Model Summary"):
                    st.text(result.summary())
                
                # Display diagnostic plots
                with st.expander("Model Diagnostics"):
                    fig_diag = result.plot_diagnostics(figsize=(15, 12))
                    st.pyplot(fig_diag)
            
            except Exception as e:
                st.error(f"Error during SARIMAX modeling: {e}")
                st.write("Please refer to the troubleshooting guide: [pmdarima Troubleshooting](http://alkaline-ml.com/pmdarima/no-successful-model.html)")
