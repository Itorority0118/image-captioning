# Image Captioning: EfficientNet-B0 + GPT-2

Dataset: Flickr30k. Encoder: EfficientNet-B0. Decoder: GPT-2.

## Chạy thử
Chạy local thì tạo môi trường ảo trước

    python -m venv .venv
    .venv\Scripts\activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

Chạy Colab, Kaggle
    pip install -r requirements.txt

Kiểm tra
    python -c "from src.paths import env; print(env)"

ai chạy local thì nó in local, colab thì colab, kaggle thì kaggle