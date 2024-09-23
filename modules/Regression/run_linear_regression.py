# modules/Regression/run_linear_regression.py
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def run_linear_regression(df):
    st.subheader("Regression Model Builder")
    
    # 1. Data Overview
    st.subheader("Data Overview")

    # Create a DataFrame with data types and missing values
    data_overview = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Values": df.isnull().sum()
    })
    st.write(data_overview)

    # 2. Select Label Columns (Not Used in Training)
    st.subheader("Select Label Columns (Not Used in Training)")
    label_columns = st.multiselect("Select Label Columns", df.columns.tolist(), key="lr_label_columns")

    # 3. Select Target Variable
    st.subheader("Select Target Variable")
    target_column = st.selectbox("Select Target Column", [col for col in df.columns if col not in label_columns], key="lr_target_column")

    # Set up feature columns
    feature_columns = [col for col in df.columns if col not in label_columns + [target_column]]

    # 4. Split into Features and Target
    X = df[feature_columns]
    y = df[target_column]

    # Handle missing values (customize as needed)
    data = pd.concat([X, y], axis=1).dropna()
    X = data[feature_columns]
    y = data[target_column]

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)

    # 5. K-Fold Cross-Validation
    st.subheader("K-Fold Cross-Validation")
    n_splits = st.slider("Select Number of Folds", min_value=2, max_value=10, value=5, key="lr_n_splits")

    # 6. Model Selection
    st.subheader("Select Regression Model")
    model_options = ["Linear Regression", "Lasso Regression (with CV)", "Ridge Regression (with CV)", "ElasticNet Regression (with CV)"]
    selected_model = st.selectbox("Choose a model", model_options, key="lr_model_selection")

    # 7. Model Training and Evaluation
    st.subheader("Model Training and Evaluation")
    if st.button("Run Model", key="lr_run_model_button"):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        st.write("Data split into training and testing sets.")

        # Initialize the model
        if selected_model == "Linear Regression":
            model = LinearRegression()
        elif selected_model == "Lasso Regression (with CV)":
            model = LassoCV(cv=n_splits, random_state=42)
        elif selected_model == "Ridge Regression (with CV)":
            model = RidgeCV(cv=n_splits)
        elif selected_model == "ElasticNet Regression (with CV)":
            model = ElasticNetCV(cv=n_splits, random_state=42)

        # Fit the model
        model.fit(X_train, y_train)
        st.write(f"{selected_model} has been trained.")

        # Predictions on test data
        y_pred = model.predict(X_test)

        # 8. Coefficients and Statistical Significance
        st.subheader("Coefficients and Statistical Significance")

        if selected_model == "Linear Regression":
            # For OLS, we can get the summary with p-values
            X_train_sm = sm.add_constant(X_train)
            ols_model = sm.OLS(y_train, X_train_sm).fit()
            st.write(ols_model.summary())
        else:
            # For regularized models, display coefficients
            coef_df = pd.DataFrame({
                "Feature": feature_columns,
                "Coefficient": model.coef_
            })
            st.write("Optimal Alpha Value:", model.alpha_ if hasattr(model, 'alpha_') else "N/A")
            st.write(coef_df)
            st.write("Note: Statistical significance (p-values) is not readily available for regularized models.")

        # 9. Residual Analysis
        st.subheader("Residual Analysis")

        residuals = y_test - y_pred

        # Residual Plot
        fig, ax = plt.subplots()
        sns.residplot(x=y_pred, y=residuals, lowess=True, ax=ax,
                      scatter_kws={'alpha': 0.5, 'color': 'indianred'}, line_kws={'color': 'orangered'})
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        st.pyplot(fig)
        plt.close(fig)

        # 10. Model Results Plot
        st.subheader("Model Results Plot")

        # Actual vs Predicted Scatter Plot
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='indianred', alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='orangered')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted (Scatter Plot)')
        st.pyplot(fig)
        plt.close(fig)

        # Actual vs Predicted Line Plot
        y_test_sorted = y_test.reset_index(drop=True)
        y_pred_series = pd.Series(y_pred, index=y_test_sorted.index)
        fig, ax = plt.subplots()
        ax.plot(y_test_sorted, label='Actual Values', color='indianred')
        ax.plot(y_pred_series, label='Predicted Values', color='orangered')
        ax.set_xlabel('Samples')
        ax.set_ylabel(target_column)
        ax.set_title('Actual vs Predicted Comparison (Line Plot)')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        # 11. Display Evaluation Metrics
        st.subheader("Evaluation Metrics")
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics_df = pd.DataFrame({
            "Metric": ["MSE", "RMSE", "MAE", "R-squared (R²)"],
            "Value": [mse, rmse, mae, r2]
        })
        st.write(metrics_df)

        # 12. Cross-Validation Scores
        st.subheader("Cross-Validation Scores")
        cv_scores = cross_val_score(
            model, X_scaled, y, 
            cv=KFold(n_splits=n_splits, shuffle=True, random_state=42), 
            scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores)
        st.write(f"**{n_splits}-Fold Cross-Validation RMSE:** {cv_rmse}")
        st.write(f"**Average CV RMSE:** {cv_rmse.mean():.4f}")

        # 13. Feature Importance (For Models that Support It)
        if hasattr(model, 'coef_'):
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Coefficient': model.coef_
            }).sort_values(by='Coefficient', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Coefficient', y='Feature', data=feature_importance, palette='viridis')
            plt.title("Feature Importance")
            st.pyplot()
            plt.close(fig)

        st.write("Model training and evaluation completed.")
