# ベースイメージとしてPythonを使用
FROM python:3.9-slim

# 必要なビルドツールと依存関係をインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    pkg-config \
    libhdf5-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを作成
WORKDIR /app

# 必要な依存関係をインストールするためにrequirements.txtをコピー
COPY requirements.txt .

# 必要なPythonパッケージをインストール
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# アプリケーションファイル全体をコンテナ内にコピー
COPY . .

# ポートを明示（任意）
EXPOSE 8501

# Streamlitアプリを起動
CMD ["streamlit", "run", "main.py", "--server.port=$PORT", "--server.address=0.0.0.0", "--server.headless=true"]
