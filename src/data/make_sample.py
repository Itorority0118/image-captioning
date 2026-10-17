import os
import random
import shutil

CSV_PATH = r"D:\flickr-30k\flickr30k_images\results.csv"
IMAGE_DIR = r"D:\flickr-30k\flickr30k_images\flickr30k_images"
NUM_IMAGES = 300

SEED = 42
OUT_DIR = "data"
OUT_IMAGE_DIR = os.path.join(OUT_DIR, "flickr30k_images")
OUT_CSV = os.path.join(OUT_DIR, "results.csv")

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)

captions = {}
names = []

f = open(CSV_PATH, encoding="utf-8")
for line in f:
    line = line.strip()
    parts = line.split("|", 2)
    if len(parts) < 3:
        continue
    name = parts[0].strip()
    number = parts[1].strip()
    text = parts[2].strip()
    if name == "image_name":
        continue
    if name not in captions:
        captions[name] = []
        names.append(name)
    captions[name].append((number, text))
f.close()

print("Tổng số ảnh trong file caption:", len(names))

random.seed(SEED)
random.shuffle(names)

out = open(OUT_CSV, "w", encoding="utf-8")
out.write("image_name| comment_number| comment\n")

count = 0
for name in names:
    if count >= NUM_IMAGES:
        break
    if len(captions[name]) < 5:
        continue
    src = os.path.join(IMAGE_DIR, name)
    if not os.path.exists(src):
        continue

    shutil.copy(src, os.path.join(OUT_IMAGE_DIR, name))
    for item in captions[name]:
        out.write(name + "| " + item[0] + "| " + item[1] + "\n")
    count = count + 1

out.close()
print("Đã lấy", count, "ảnh vào thư mục", OUT_DIR)