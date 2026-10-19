import os
import sys
from PIL import Image

IMAGE_DIR = os.path.join("data", "flickr30k_images")
CSV_PATH = os.path.join("data", "results.csv")
REPORT_PATH = os.path.join("outputs", "image_report.txt")

if len(sys.argv) >= 2:
    IMAGE_DIR = sys.argv[1]
if len(sys.argv) >= 3:
    CSV_PATH = sys.argv[2]

def read_caption_names(csv_path):
    names = set()
    f = open(csv_path, encoding="utf-8")
    for line in f:
        parts = line.strip().split("|", 2)
        if len(parts) < 3:
            continue
        name = parts[0].strip()
        if name == "image_name":
            continue
        names.add(name)
    f.close()
    return names

files = []
for name in os.listdir(IMAGE_DIR):
    if name.lower().endswith(".jpg"):
        files.append(name)
files.sort()
print("Số ảnh tìm thấy:", len(files))

widths = []
heights = []
size_count = {}
bad_files = []
not_rgb = 0

for name in files:
    path = os.path.join(IMAGE_DIR, name)
    try:
        img = Image.open(path)
        w, h = img.size
        img.load()
        if img.mode != "RGB":
            not_rgb = not_rgb + 1
        img.close()
    except Exception as e:
        bad_files.append(name)
        continue
    widths.append(w)
    heights.append(h)
    key = str(w) + "x" + str(h)
    if key not in size_count:
        size_count[key] = 0
    size_count[key] = size_count[key] + 1

caption_names = read_caption_names(CSV_PATH)
file_set = set(files)
missing_image = 0
for name in caption_names:
    if name not in file_set:
        missing_image = missing_image + 1
no_caption = 0
for name in files:
    if name not in caption_names:
        no_caption = no_caption + 1

lines = []
ok_count = len(widths)
lines.append("Thư mục ảnh: " + IMAGE_DIR)
lines.append("Số file ảnh: " + str(len(files)))
lines.append("Mở được: " + str(ok_count))
lines.append("Ảnh lỗi (không mở được): " + str(len(bad_files)))
if ok_count > 0:
    lines.append("Chiều rộng:  nhỏ nhất " + str(min(widths)) + ", lớn nhất " + str(max(widths)) + ", trung bình " + str(round(sum(widths) / ok_count)))
    lines.append("Chiều cao:   nhỏ nhất " + str(min(heights)) + ", lớn nhất " + str(max(heights)) + ", trung bình " + str(round(sum(heights) / ok_count)))
lines.append("Ảnh không phải RGB (cần đổi sang RGB): " + str(not_rgb))
lines.append("Số tên ảnh có trong file caption: " + str(len(caption_names)))
lines.append("Có trong caption nhưng thiếu file ảnh: " + str(missing_image))
lines.append("Có file ảnh nhưng không có caption: " + str(no_caption))

pairs = []
for key in size_count:
    pairs.append((size_count[key], key))
pairs.sort(reverse=True)
lines.append("5 kích thước phổ biến nhất:")
top = 0
for p in pairs:
    if top >= 5:
        break
    lines.append("  " + p[1] + ": " + str(p[0]) + " ảnh")
    top = top + 1

if len(bad_files) > 0:
    lines.append("Danh sách ảnh lỗi:")
    for name in bad_files:
        lines.append("  " + name)

for line in lines:
    print(line)

os.makedirs("outputs", exist_ok=True)
f = open(REPORT_PATH, "w", encoding="utf-8")
for line in lines:
    f.write(line + "\n")
f.close()
print("Đã lưu báo cáo:", REPORT_PATH)