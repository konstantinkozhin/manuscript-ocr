import json

# 1) Загрузим оригинальный JSON
with open(r"C:\data0205\Archives020525\train.json", "r", encoding="utf-8") as f:
    coco = json.load(f)

# 2) Проходим по аннотациям, собираем только первые вхождения каждого bbox
seen = set()
clean_anns = []
for ann in coco["annotations"]:
    if ann["category_id"] != 0:  # ваша категория
        clean_anns.append(ann)
        continue
    bbox_key = tuple(ann["bbox"])
    if bbox_key not in seen:
        seen.add(bbox_key)
        clean_anns.append(ann)

print("Оригинальных anns:", len(coco["annotations"]))
print("GT-category anns:", sum(1 for a in coco["annotations"] if a["category_id"] == 0))
print("Уникальных bbox:", len(seen))
print("Аннотаций после очистки:", len(clean_anns))

# 3) Заменим секцию annotations и сохраним в новый файл
coco_clean = coco.copy()
coco_clean["annotations"] = clean_anns

with open(r"C:\data0205\Archives020525\train2.json", "w", encoding="utf-8") as f:
    json.dump(coco_clean, f, ensure_ascii=False, indent=2)

print("Сохранили очищенный JSON в test_clean.json")
