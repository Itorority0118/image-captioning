import json
import os
import random
import sys

SEED = 42
NUM_CAPTIONS = 1  
FULL_VAL = 1000       
FULL_TEST = 1000   
SAMPLE_VAL_RATIO = 0.1  
SAMPLE_TEST_RATIO = 0.1

SAMPLE_CSV = os.path.join("data", "results.csv")
OUT_FULL = os.path.join("configs", "splits.json")
OUT_SAMPLE = os.path.join("configs", "splits_sample.json")

def read_image_names(csv_path):
    count = {}
    order = []

    f = open(csv_path, encoding="utf-8")
    for line in f:
        line = line.strip()
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        name = parts[0].strip()
        if name == "image_name":
            continue
        if name not in count:
            count[name] = 0
            order.append(name)
        count[name] = count[name] + 1
    f.close()

    names = []
    skipped = 0
    for name in order:
        if count[name] >= NUM_CAPTIONS:
            names.append(name)
        else:
            skipped = skipped + 1

    print("Total:", len(order))
    print("Skipped: ", skipped)
    print("Remain:", len(names))
    return names


def make_split(names, num_val, num_test):
    names = sorted(names)
    random.seed(SEED)
    random.shuffle(names)

    val = names[:num_val]
    test = names[num_val:num_val + num_test]
    train = names[num_val + num_test:]
    return train, val, test

def check_no_overlap(train, val, test):
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)
    for name in val:
        if name in train_set:
            print("LỖI: ảnh vừa ở val vừa ở train:", name)
            sys.exit(1)
    for name in test:
        if name in train_set or name in val_set:
            print("LỖI: ảnh ở test trùng với tập khác:", name)
            sys.exit(1)
    print("OK")

def save_json(train, val, test, out_path):
    folder = os.path.dirname(out_path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    result = {"train": train, "val": val, "test": test}
    f = open(out_path, "w", encoding="utf-8")
    json.dump(result, f, ensure_ascii=False)
    f.close()
    print("Đã lưu:", out_path)
    print("train:", len(train), "| val:", len(val), "| test:", len(test))

def main():
    mode = sys.argv[1]

    if mode == "sample":
        names = read_image_names(SAMPLE_CSV)
        num_val = int(len(names) * SAMPLE_VAL_RATIO)
        num_test = int(len(names) * SAMPLE_TEST_RATIO)
        out_path = OUT_SAMPLE
    elif mode == "full":
        names = read_image_names(sys.argv[2])
        num_val = FULL_VAL
        num_test = FULL_TEST
        out_path = OUT_FULL

    if len(names) <= num_val + num_test:
        print("Số ảnh (" + str(len(names)) + ") không đủ để chia. Cần nhiều hơn", num_val + num_test, "ảnh.")
        sys.exit(1)

    train, val, test = make_split(names, num_val, num_test)
    check_no_overlap(train, val, test)
    save_json(train, val, test, out_path)

main()