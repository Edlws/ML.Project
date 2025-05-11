from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_from_disk
from seqeval.metrics import classification_report
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import json

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    true_labels = []
    true_preds = []

    for pred, label in zip(predictions, labels):
        true_labels_seq = []
        true_preds_seq = []
        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                true_labels_seq.append(id2label[l_i])
                true_preds_seq.append(id2label[p_i])
        true_labels.append(true_labels_seq)
        true_preds.append(true_preds_seq)

    print("\n" + classification_report(true_labels, true_preds))

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds)
    }



label_list = ["O", "B-PRODUCT", "I-PRODUCT"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

dataset = load_from_disk("product_ner_dataset")

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    output_dir="./ner_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./ner_model")
tokenizer.save_pretrained("./ner_model")

metrics = trainer.evaluate()
print("Final eval metrics:", metrics)

with open("eval_metrics.json", "w") as f:
    json.dump(metrics, f)




