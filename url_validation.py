import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def is_html(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
            soup = BeautifulSoup(r.text, "html.parser")
            title = soup.title.string.strip() if soup.title else "No title"
            return {"url": url, "html": r.text, "title": title}
    except Exception:
        return None

def process_csv(input_csv, output_jsonl="valid_urls.jsonl", max_threads=20):
    df = pd.read_csv(input_csv)
    urls = df['max(page)'].dropna().unique().tolist()

    results = []

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(is_html, url): url for url in urls}

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                results.append(result)

    with open(output_jsonl, "w", encoding="utf-8") as out:
        for item in results:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    process_csv("URL_list.csv")
