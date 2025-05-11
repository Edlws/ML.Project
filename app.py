import streamlit as st
from transformers import pipeline
from boilerpy3 import extractors
import json
import os

extractor = extractors.ArticleExtractor()

st.sidebar.header("Model metrics")

if os.path.exists("eval_metrics.json"):
    with open("eval_metrics.json", "r") as f:
        metrics = json.load(f)

    st.sidebar.metric("F1-score", f"{metrics['eval_f1']:.2f}")
    st.sidebar.metric("Precision", f"{metrics['eval_precision']:.2f}")
    st.sidebar.metric("Recall", f"{metrics['eval_recall']:.2f}")

if os.path.exists("eval_report.txt"):
    with open("eval_report.txt", "r") as f:
        report = f.read()
    with st.expander("Report: (classification_report)"):
        st.text(report)

@st.cache_resource
def load_model():
    return pipeline("ner", model="./ner_model", tokenizer="./ner_model", aggregation_strategy="simple")

ner = load_model()

st.title("Product Extractor")

url = st.text_input("Enter URL:")

if url:
    st.write("Processing...")
    
    try:

        downloaded = extractor.get_doc_from_url(url)
        
        if downloaded:
            text = downloaded.content
            if text:
                st.success("Text successfully extracted")
                with st.expander("Show text"):
                    st.write(text) 

                st.write("Searching for products...")
                results = ner(text)
                products = [r["word"] for r in results if r["entity_group"] == "PRODUCT"]

                if products:
                    st.success("Products found:")
                    for p in set(products):
                        st.write(f"• {p}")
                else:
                    st.warning("No products found.")
            else:
                st.error("Failed to extract text from the page")
        else:
            st.error("Failed to download the page content")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")