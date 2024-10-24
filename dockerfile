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

# ポートを明示（Cloud Runは8080を期待）
EXPOSE 8080

# Streamlitアプリを起動（シェル形式で$PORTを展開）
CMD ["sh", "-c", "streamlit run main.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true"]
