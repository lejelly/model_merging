# NVIDIA CUDA 12.2 をベースイメージとして使用
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3.10-venv \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*


# pipのアップグレード
RUN pip3 install --upgrade pip

# PyTorchのインストール
RUN pip3 install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 作業ディレクトリの設定
WORKDIR /workspace

# コンテナ起動時のコマンド
CMD ["/bin/bash"]