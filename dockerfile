# ベースイメージとしてPython 3.9の軽量バージョンを使用
FROM python:3.9-slim-buster

# システム依存パッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# requirements.txtをコピー
COPY requirements.txt /app/

# Pythonパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードと必要なファイルをコピー
COPY . /app

# ポートを公開（Cloud Runはデフォルトで8080ポートを使用）
EXPOSE 8080

# 環境変数の設定
ENV PORT 8080
ENV STREAMLIT_SERVER_PORT $PORT
ENV STREAMLIT_SERVER_ADDRESS 0.0.0.0
ENV STREAMLIT_SERVER_ENABLE_CORS false

# アプリケーションの起動コマンド
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
