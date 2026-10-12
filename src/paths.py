import os
from pathlib import Path

def get_env():
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    if os.path.exists("/content"):
        return "colab"
    return "local"

ENV = get_env()
if ENV == "kaggle":
    DATA_PATH = Path("/kaggle/input/flickr30k")
    WORK_PATH = Path("/kaggle/working")
elif ENV == "colab":
    DATA_PATH = Path("/content/data")
    WORK_PATH = Path("/content/work")
else:
    DATA_PATH = Path(os.getenv("DATA_PATH", "./data"))
    WORK_PATH = Path(os.getenv("WORK_PATH", "./outputs"))
feature_path = WORK_PATH / "features"
checkpoint_path = WORK_PATH / "checkpoints"
for d in (feature_path, checkpoint_path):
    d.mkdir(parents=True, exist_ok=True)