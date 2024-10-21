import streamlit as st
import pandas as pd

# 各モジュールのインポート
from modules.EDA_Options.show_summary import show_summary
from modules.EDA_Options.show_histogram import show_histogram
from modules.EDA_Options.show_crosstab_plot import show_crosstab_plot

from modules.Feature_Engineering.show_dataframe_info import show_dataframe_info
from modules.Feature_Engineering.start_feature_engineering import start_feature_engineering

from modules.Binary_Classification.run_lightgbm import run_lightgbm
from modules.Binary_Classification.run_catboost import run_catboost

from modules.Regression.run_linear_regression import run_linear_regression

from modules.Effectiveness_Verification.show_histogram_treatment import show_histogram_treatment
from modules.Effectiveness_Verification.run_bayesian_ab_test import run_bayesian_ab_test
from modules.Effectiveness_Verification.run_t_statistic_ab_test import run_t_statistic_ab_test

from modules.Time_Series_Forecasting.initial_setting_quick_eda import initial_setting_quick_eda
from modules.Time_Series_Forecasting.run_stl_adf_autocorrelation import run_stl_adf_autocorrelation
from modules.Time_Series_Forecasting.run_sarimax import run_sarimax
from modules.Time_Series_Forecasting.run_prophet import run_prophet

# カスタムCSSを追加
st.markdown(
    """
    <style>
    /* サイドバーの背景色を変更 */
    .css-1d391kg {
        background-color: indianred;  /* ここで背景色を変更します */
    }
    
    /* サイドバーの文字色を変更 */
    .css-1d391kg, .css-1v3fvcr {
        color: white;  /* ここで文字色を変更します */
    }

    /* スクロールバーの色も変更可能 */
    .css-1v3fvcr::-webkit-scrollbar {
        background-color: #2E8B57;  /* スクロールバーの背景色 */
        width: 5px; /* スクロールバーの幅 */
    }

    .css-1v3fvcr::-webkit-scrollbar-thumb {
        background-color: #4CAF50; /* スクロールバーのハンドル部分の色 */
    }

    </style>
    """,
    unsafe_allow_html=True
)

# プログレスバーのカスタムスタイルを適用
st.markdown(
    """
    <style>
    .stProgress > div > div > div > div {
        background-color: OrangeRed;
    }
    </style>""",
    unsafe_allow_html=True
)

# SVG画像の表示（パスは適宜変更してください）
svg_file = 'statisticsanalysisapps.svg'
st.image(svg_file, width=700)

# デモ用のデータセット
demo_datasets = {
    "Iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "EffectivenessData" : "lenta_dataset.csv",
    "COVID-19" : "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv",
    "sales_data_with_attributes" : "sales_data_with_attributes.csv"
}

# サイドバーにファイルアップロードウィジェットを配置
st.sidebar.header("Upload or select train data")
uploaded_train_file = st.sidebar.file_uploader("Drag and drop file here", type=["csv", "xlsx"])

# デモデータセットの選択
selected_demo_dataset = st.sidebar.selectbox("Or select a demo dataset", list(demo_datasets.keys()))

if uploaded_train_file is not None:
    # ファイルの読み込み（CSVまたはExcel）
    if uploaded_train_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_train_file)
    else:
        df = pd.read_excel(uploaded_train_file)
    source = "Uploaded File"
else:
    # デモデータセットの読み込み
    try:
        df = pd.read_csv(demo_datasets[selected_demo_dataset])
        source = f"Demo Dataset ({selected_demo_dataset})"
    except FileNotFoundError:
        st.error(f"Demo dataset '{selected_demo_dataset}' not found. Please upload your own data.")
        st.stop()

# データフレームの表示
if st.sidebar.button("Make DataFrame"):
    st.subheader(f'Data Preview - {source}')
    st.text("Numbers of Columns & Rows")
    st.write(f'Columns : {df.shape[1]}, Rows : {df.shape[0]}')
    st.text("ーーーーーーーーーーーーーーーーーーーーーーー")
    st.text("20 appear")
    st.dataframe(df.head(20))

if st.sidebar.button("Refresh DataFrame"):
    # セッションステートをクリアしてリフレッシュに相当する動作をする
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

# サイドバーにEDAオプションを配置
st.sidebar.subheader("1. EDA Options")

# Show Summary
if st.sidebar.checkbox("Show Summary"):
    show_summary(df)

# Show Histogram
if st.sidebar.checkbox("Show Histogram "):
    show_histogram(df)

# Show Crosstab Plot
if st.sidebar.checkbox("Show Crosstab Plot"):
    show_crosstab_plot(df)

# Feature Engineering
st.sidebar.subheader("2. Feature Engineering")

if st.sidebar.checkbox("Show DataFrame Infomation"):
    show_dataframe_info(df)

if st.sidebar.checkbox("Start Feature Engineering"):
    start_feature_engineering(df)

# Binary Classification
st.sidebar.subheader("3. Binary Classification")

if st.sidebar.checkbox("Run LightGBM"):
    run_lightgbm(df)

if st.sidebar.checkbox("Run CatBoost"):
    run_catboost(df)

# Regression
st.sidebar.subheader("4. Regression")

if st.sidebar.checkbox("Run Linear Regression"):
    run_linear_regression(df)

# Effectiveness Verification
st.sidebar.subheader("5. Effectiveness Verification")

if st.sidebar.checkbox("Show histogram Treatment & Others"):
    show_histogram_treatment(df)

if st.sidebar.checkbox("Run Bayesian AB Test"):
    run_bayesian_ab_test(df)

if st.sidebar.checkbox("Run T-Statistic AB Test"):
    run_t_statistic_ab_test(df)

# Time Series Forecasting
st.sidebar.subheader("6. Time Series Forecasting")

if st.sidebar.checkbox("InitialSetting & QuickEDA"):  
    initial_setting_quick_eda(df)

if st.sidebar.checkbox("STL/ADF/Autocorrelation"):  
    run_stl_adf_autocorrelation(df)

if st.sidebar.checkbox("Run SARIMAX"):
    run_sarimax(df)

if st.sidebar.checkbox("Run Prophet"):
    run_prophet(df)
