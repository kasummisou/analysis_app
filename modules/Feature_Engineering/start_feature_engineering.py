# modules/Feature_Engineering/start_feature_engineering.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt

def start_feature_engineering(df):
    # 欠損値処理
    st.header("Show Missing Value")

    # データフレームの表示
    st.write("Show Summary of Missing Values:")
    summary_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.values,
        "Missing Values": df.isnull().sum().values,
        "Number Of '0' Values": (df == 0).sum().values,
        "Unique Values": df.nunique().values
    })
    st.dataframe(summary_df)

    # 欠損値を持つカラムを特定
    missing_columns = df.columns[df.isnull().any()].tolist()

    # 全てのカラムを選択肢として提供
    all_columns = df.columns.tolist()

    # 欠損値のあるカラムを表示
    st.write("欠損値を持つカラム:")
    st.write(missing_columns)

    if missing_columns:
        # X軸とY軸に使用するカラムを選択
        x_column = st.selectbox('Select a column for the X-axis', missing_columns, key="fe_x_axis")
        y_column = st.selectbox('Select a column for the Y-axis', all_columns, key="fe_y_axis")

        if x_column and y_column:
            # 欠損値の有無を示すデータフレームを作成
            missing_data = df.isnull()

            # X軸とY軸に使用するカラムの値ごとに欠損値の割合を計算
            combined_data = df[[x_column, y_column]].copy()
            combined_data['missing'] = missing_data.sum(axis=1) > 0  # 欠損値の有無

            # ピボットテーブルを作成
            heatmap_data = combined_data.pivot_table(index=y_column, columns=x_column, values='missing', aggfunc='mean')

            # カスタムカラーマップを作成
            cmap = LinearSegmentedColormap.from_list('custom_red_gradient', ['orange', 'orangered', 'indianred'])

            # 欠損値のヒートマップ
            st.write("Heatmap of Missing:")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(heatmap_data, cmap=cmap, cbar=True, ax=ax)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            st.pyplot(fig)

    st.header("Missing Value Handling")
    
    missing_action = st.radio("Select missing value handling method", ('Fill with mean', 'Drop rows with missing values'), key="fe_missing_action")

    process_missing_values = st.button("Start Processing", key="fe_start_processing")

    if process_missing_values:
        if missing_action == 'Fill with mean':
            df.fillna(df.mean(numeric_only=True), inplace=True)
            st.write("DataFrame after filling missing values with mean:")
            st.write(df)
        elif missing_action == 'Drop rows with missing values':
            df.dropna(inplace=True)
            st.write("DataFrame after dropping rows with missing values:")
            st.write(df)

    # オブジェクト型または文字列型のカラムを選択する
    st.header("Encoding Value Handling")
    object_columns = df.select_dtypes(include=['object', 'string']).columns
    if len(object_columns) > 0:
        selected_column = st.selectbox("Select a column for encoding", object_columns, key="fe_encoding_column")
    
        if selected_column:
            encoding_method = st.radio("Select encoding method", ('One-hot encoding', 'Label encoding'), key="fe_encoding_method")

            process_encoding = st.button("Start Encoding", key="fe_start_encoding")

            if process_encoding:
                if encoding_method == 'One-hot encoding':
                    df = pd.get_dummies(df, columns=[selected_column])
                    st.write("One-hot encoded DataFrame:")
                    st.write(df)
                elif encoding_method == 'Label encoding':
                    label_encoder = LabelEncoder()
                    df[selected_column + '_encoded'] = label_encoder.fit_transform(df[selected_column])
                    st.write("Label encoded DataFrame:")
                    st.write(df)
    else:
        st.write("No object or string type columns available for encoding.")

    # CSVファイルとしてダウンロードする機能
    st.header("Download Processed Data")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='processed_data.csv',
        mime='text/csv',
    )
