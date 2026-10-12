import os
from pathlib import Path

def get_env():
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    if os.path.exists("/content"):
        return "colab"
    return "local"

env = get_env()
if env == "kaggle":
    data_path = Path("/kaggle/input/flickr30k")
    work_path = Path("/kaggle/working")
elif env == "colab":
    data_path = Path("/content/data")
    work_path = Path("/content/work")
else:
    data_path = Path(os.getenv("data_path", "./data"))
    work_path = Path(os.getenv("work_path", "./outputs"))
feature_path = work_path / "features"
checkpoint_path = work_path / "checkpoints"
for d in (feature_path, checkpoint_path):
    d.mkdir(parents=True, exist_ok=True)