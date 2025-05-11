import json
from datasets import Dataset

def read_doccano(file_path):
    with open(file_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def tokenize_and_align_labels(text, entities):
    tokens = text.split()
    token_labels = ["O"] * len(tokens)

    for start, end, label in entities:
        first_token_found = False
        for i, token in enumerate(tokens):
            token_start = text.find(token, 0 if i == 0 else text.find(tokens[i - 1]) + len(tokens[i - 1]))
            token_end = token_start + len(token)
            if token_start >= start and token_end <= end:
                if not first_token_found:
                    token_labels[i] = "B-" + label
                    first_token_found = True
                else:
                    token_labels[i] = "I-" + label
    return tokens, token_labels

def convert_to_dataset(doccano_data):
    all_data = {"tokens": [], "ner_tags": []}

    for item in doccano_data:
        text = item["text"]
        entities = [(e[0], e[1], e[2]) for e in item["label"]]
        tokens, labels = tokenize_and_align_labels(text, entities)

        all_data["tokens"].append(tokens)
        all_data["ner_tags"].append(labels)

    return Dataset.from_dict(all_data)

if __name__ == "__main__":
    data = read_doccano("admin.jsonl")
    dataset = convert_to_dataset(data)
    dataset.save_to_disk("product_ner_dataset")
