
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px

# Streamlit Cheat Sheet

# 1. Basic Usage

# 1.1 Text Display
st.title('タイトル')
st.header('ヘッダー')
st.subheader('サブヘッダー')
st.text('テキストを表示')
st.markdown('# Markdownを使用')
st.write('オブジェクトやテキストを表示')

# 1.2 Display DataFrame
df = pd.DataFrame({
    '列1': [1, 2, 3, 4],
    '列2': [10, 20, 30, 40]
})

st.dataframe(df)
st.table(df)

# 1.3 Display Charts

# Matplotlib Chart
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 30, 40])
st.pyplot(fig)

# Seaborn Chart
sns_plot = sns.scatterplot(x=[1, 2, 3, 4], y=[10, 20, 30, 40])
st.pyplot(sns_plot.figure)

# Plotly Chart
fig = px.line(x=[1, 2, 3, 4], y=[10, 20, 30, 40])
st.plotly_chart(fig)

# 2. User Input

# 2.1 Input Widgets
name = st.text_input('名前を入力してください')
age = st.number_input('年齢を入力してください', min_value=0, max_value=100)
value = st.slider('スライダー', min_value=0, max_value=100, value=50)

# 2.2 Select Box
option = st.selectbox('選択してください', ['オプション1', 'オプション2', 'オプション3'])

# 2.3 Multi-select
options = st.multiselect('複数選択', ['オプションA', 'オプションB', 'オプションC'])

# 2.4 Check Box
if st.checkbox('チェックを入れる'):
    st.write('チェックが入りました')

# 2.5 Radio Button
status = st.radio('ステータスを選択', ['Active', 'Inactive'])

# 2.6 Button
if st.button('クリック'):
    st.write('ボタンがクリックされました')

# 2.7 File Upload
uploaded_file = st.file_uploader("ファイルをアップロード", type=["csv", "xlsx"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

# 3. Layout

# 3.1 Sidebar
st.sidebar.title('サイドバー')
side_option = st.sidebar.selectbox('サイドバーのセレクトボックス', ['オプション1', 'オプション2'])

# 3.2 Column Layout
col1, col2 = st.columns(2)

with col1:
    st.write('左側のカラム')

with col2:
    st.write('右側のカラム')

# 3.3 Expander
with st.expander("クリックして展開"):
    st.write("折りたたみメニューの中身")

# 4. State Management

# 4.1 Session State
if 'count' not in st.session_state:
    st.session_state.count = 0

increment = st.button('インクリメント')
if increment:
    st.session_state.count += 1

st.write('カウント:', st.session_state.count)

# 5. Media Display

# 5.1 Display Image
image = Image.open('path_to_image.png')
st.image(image, caption='キャプション付き画像')

# 5.2 Display Video
st.video('path_to_video.mp4')

# 5.3 Play Audio
st.audio('path_to_audio.mp3')

# 6. Download

# 6.1 Download Data
csv = df.to_csv().encode('utf-8')
st.download_button("CSVをダウンロード", csv, "file.csv", "text/csv", key='download-csv')

# 7. Deployment

# 7.1 Deploy on Streamlit Cloud
# - Push your code to a GitHub repository.
# - Go to Streamlit Cloud and deploy your app from the repository.
