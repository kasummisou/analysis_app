# ベースイメージとしてPythonを使用
FROM python:3.9-slim

# 必要なビルドツールと依存関係をインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    pkg-config \
    libhdf5-dev \
    && apt-get clean

# 作業ディレクトリを作成
WORKDIR /app

# 必要な依存関係をインストールするためにrequirements.txtをコピー
COPY requirements.txt .

# 必要なPythonパッケージをインストール
RUN pip install --upgrade pip && pip install -r requirements.txt

# アプリケーションファイル全体をコンテナ内にコピー
COPY . .

# Streamlitアプリを起動
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.headless=true"]