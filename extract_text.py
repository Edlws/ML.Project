import json
import trafilatura
from tqdm import tqdm

def load_urls(jsonl_path, limit=270):
    urls = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(urls) >= limit:
                break
            try:
                data = json.loads(line)
                urls.append(data['url'])
            except:
                continue
    return urls

def extract_and_save(urls, output_path):
    with open(output_path, 'w', encoding='utf-8') as fout:
        for url in tqdm(urls, desc="Extracting"):
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                continue
            text = trafilatura.extract(downloaded)
            if not text:
                continue
            json_line = json.dumps({
                "url": url,
                "text": text.strip()
            }, ensure_ascii=False)
            fout.write(json_line + "\n")

if __name__ == "__main__":
    input_jsonl = "valid_urls.jsonl"
    output_jsonl = "clean_texts.json"

    urls = load_urls(input_jsonl, limit=270)
    extract_and_save(urls, output_jsonl)
