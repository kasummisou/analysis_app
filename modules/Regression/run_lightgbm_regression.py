# modules/Regression/run_lightgbm_regression.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

def run_lightgbm_regression(df):
    st.subheader("LightGBM Regression Model Builder")
    
    # 1. Data Overview
    st.subheader("1. Data Overview")
    
    data_overview = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Values": df.isnull().sum()
    })
    st.write(data_overview)
    
    # 2. Select Label Columns (Not Used in Training)
    st.subheader("2. Select Label Columns (Not Used in Training)")
    label_columns = st.multiselect("Select Label Columns to Exclude from Training", df.columns.tolist(), key="lgbm_reg_label_columns")
    
    # 3. Select Target Variable
    st.subheader("3. Select Target Variable")
    target_column = st.selectbox("Select Target Column", [col for col in df.columns if col not in label_columns], key="lgbm_reg_target_column")
    
    # 4. Set Feature Columns
    feature_columns = [col for col in df.columns if col not in label_columns + [target_column]]
    
    # 5. Split into Features and Target
    X = df[feature_columns]
    y = df[target_column]
    
    # 6. Handle Missing Values (Here, dropping rows with missing values)
    data = pd.concat([X, y], axis=1).dropna()
    X = data[feature_columns]
    y = data[target_column]
    st.write(f"Number of rows after dropping rows with missing values: {data.shape[0]} rows")
    
    # 7. Data Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
    
    # 8. K-Fold Cross-Validation Settings
    st.subheader("4. K-Fold Cross-Validation Settings")
    n_splits = st.slider("Select Number of Folds", min_value=2, max_value=10, value=5, key="lgbm_reg_n_splits")
    
    # 9. Select Evaluation Metric for Hyperparameter Tuning
    st.subheader("5. Select Evaluation Metric for Hyperparameter Tuning")
    evaluation_metric = st.selectbox(
        "Choose Evaluation Metric",
        ("Mean Absolute Error (MAE)", "Mean Squared Error (MSE)")
    )
    
    # 10. Select Hyperparameter Tuning Method
    st.subheader("6. Select Hyperparameter Tuning Method")
    tuning_method = st.radio(
        "Choose a hyperparameter tuning method:",
        ("Optuna", "RandomizedSearchCV")
    )
    
    # Map selected metric to LightGBM and scikit-learn parameters
    if evaluation_metric == "Mean Absolute Error (MAE)":
        lgb_objective = "regression_l1"
        scoring = "neg_mean_absolute_error"
        direction = "minimize"
        metric = "l1"
    elif evaluation_metric == "Mean Squared Error (MSE)":
        lgb_objective = "regression"
        scoring = "neg_mean_squared_error"
        direction = "minimize"
        metric = "l2"
    else:
        st.error("Unsupported evaluation metric selected.")
        st.stop()

    # 11. Hyperparameter Grid for RandomizedSearchCV
    param_distributions = {
        'num_leaves': [31, 50, 70, 100],
        'max_depth': [-1, 10, 15, 20, 25],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300, 500, 1000],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_samples': [20, 30, 40, 50]
    }
    
    # 12. Model Training and Evaluation
    st.subheader("7. Model Training and Evaluation")
    if st.button("Run LightGBM Model with Hyperparameter Tuning", key="lgbm_reg_run_model_button"):
        with st.spinner('Training the model...'):
            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            if tuning_method == "RandomizedSearchCV":
                # Initialize LightGBM Regressor with selected objective
                base_model = LGBMRegressor(
                    objective=lgb_objective,
                    random_state=42
                )
                
                # Initialize RandomizedSearchCV
                random_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_distributions,
                    n_iter=50,  # Number of parameter settings sampled
                    scoring=scoring,
                    cv=KFold(n_splits=n_splits, shuffle=True, random_state=42),
                    verbose=0,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Perform Random Search
                random_search.fit(X_train, y_train)
                
                # Retrieve best parameters and score
                best_params = random_search.best_params_
                best_score = -random_search.best_score_
                
                st.write("### Best Hyperparameters Found:")
                st.write(best_params)
                st.write(f"**Best {evaluation_metric}:** {best_score:.4f}")
                
                # Train the model with best parameters
                best_model = random_search.best_estimator_
                best_model.fit(X_train, y_train)
            
            elif tuning_method == "Optuna":
                # Define the objective function for Optuna
                def objective(trial):
                    params = {
                        'num_leaves': trial.suggest_categorical('num_leaves', [31, 50, 70, 100]),
                        'max_depth': trial.suggest_categorical('max_depth', [-1, 10, 15, 20, 25]),
                        'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.05, 0.1]),
                        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 500, 1000]),
                        'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
                        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.7, 0.8, 0.9, 1.0]),
                        'min_child_samples': trial.suggest_categorical('min_child_samples', [20, 30, 40, 50])
                    }
                    model = LGBMRegressor(**params, objective=lgb_objective, random_state=42)
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
                    y_pred_proba = model.predict(X_test)
                    if evaluation_metric == "Mean Absolute Error (MAE)":
                        return mean_absolute_error(y_test, y_pred_proba)
                    elif evaluation_metric == "Mean Squared Error (MSE)":
                        return mean_squared_error(y_test, y_pred_proba)
                
                study = optuna.create_study(direction=direction)
                study.optimize(objective, n_trials=100)
                best_params = study.best_params
                st.write("### Best Hyperparameters Found:")
                st.write(best_params)
                best_model = LGBMRegressor(**best_params, objective=lgb_objective, random_state=42)
                best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
            
            # Predictions on Test Data
            y_pred = best_model.predict(X_test)
            
            # Calculate Evaluation Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.write("### Evaluation Metrics on Test Data")
            st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
            st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
            st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
            st.write(f"**R-squared (R²):** {r2:.4f}")
            
            # Cross-Validation Scores
            cv_scores = cross_val_score(
                best_model, X_scaled, y, 
                cv=KFold(n_splits=n_splits, shuffle=True, random_state=42), 
                scoring=scoring
            )
            cv_scores_positive = -cv_scores
            st.write(f"**{n_splits}-Fold Cross-Validation {evaluation_metric}:** {cv_scores_positive}")
            st.write(f"**Average Cross-Validation {evaluation_metric}:** {cv_scores_positive.mean():.4f}")
            
            # Feature Importance
            st.subheader("8. Feature Importance")
            importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': best_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance, ax=ax, palette='viridis')
            ax.set_title('Feature Importances')
            st.pyplot(fig)
            plt.close(fig)

            # Residual Analysis
            st.subheader("9. Residual Analysis")
            residuals = y_test - y_pred
            
            fig, ax = plt.subplots()
            sns.residplot(x=y_pred, y=residuals, lowess=True, ax=ax,
                          scatter_kws={'alpha': 0.5, 'color': 'indianred'}, line_kws={'color': 'orangered'})
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residual Plot")
            st.pyplot(fig)
            plt.close(fig)

            # Actual vs Predicted Comparison
            st.subheader("10. Actual vs Predicted Comparison")
            
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
