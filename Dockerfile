# ベースイメージとしてPythonを使用
FROM python:3.9-slim

# 必要なビルドツールと依存関係をインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    pkg-config \
    libhdf5-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean

# 作業ディレクトリを作成
WORKDIR /app

# 必要な依存関係をインストールするためにrequirements.txtをコピー
COPY requirements.txt .

# 必要なPythonパッケージをインストール
RUN pip install --upgrade pip && pip install -r requirements.txt

# アプリケーションファイル全体をコンテナ内にコピー
COPY . .

# Uvicornを用いてFastAPIとStreamlitを同時に起動
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8080 & streamlit run main.py --server.port=8501 --server.headless=true --server.enableCORS=false"]
