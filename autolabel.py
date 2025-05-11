import json
from transformers import pipeline
from tqdm import tqdm
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)

ner = pipeline(
    "ner",
    model="DeveloperSejin/NER_for_furniture_3D_object_create",
    aggregation_strategy="simple",
    framework="pt"
)

with open("clean_texts.jsonl", "r", encoding="utf-8") as f:
    texts = [json.loads(line)["text"] for line in f if line.strip()]

output = []

for text in tqdm(texts, desc="Annotating"):
    try:
        entities = ner(text)
        labels = []
        
        for ent in entities:
            if ent.get("entity_group", "").upper() in {"FUR", "FURNITURE", "PRODUCT"}:
                labels.append([
                    int(ent["start"]),
                    int(ent["end"]),
                    "PRODUCT"
                ])
        
        output.append({
            "text": text,
            "label": labels
        })
    except Exception as e:
        print(f"Ошибка при обработке текста: {text[:50]}... - {str(e)}")

with open("admin_auto.jsonl", "w", encoding="utf-8") as f:
    for item in output:
        f.write(json.dumps(item, ensure_ascii=False, cls=NumpyEncoder) + "\n")

