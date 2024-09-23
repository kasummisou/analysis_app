# modules/Feature_Engineering/show_dataframe_info.py

import streamlit as st
import pandas as pd

def show_dataframe_info(df):
    st.subheader("DataFrame")
    st.dataframe(df.head(10))
    
    st.subheader("DataFrame Information")
    info_df = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Values": df.isnull().sum(),
        "Number of '0' Values": (df == 0).sum(),
        "Unique Values": df.nunique()
    })
    st.dataframe(info_df)
