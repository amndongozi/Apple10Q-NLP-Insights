# %%
# CELL 1: Setup and File Paths
# This cell sets up the environment and defines file paths.
import os
import re
import pandas as pd
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define file paths using relative paths that are easy to manage
TEN_Q_FILE = "data/10Q(2017-02-01) AAPL.pdf"
GARTNER_FILE = "data/Gartner_Hype_Cycle__publicly_listed_items__2000_2025.csv"

# %%
# CELL 2: Initialize FinBERT Model
# This cell loads the FinBERT model and tokenizer for local sentiment analysis.
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
print("FinBERT model loaded successfully.")

# %%
# CELL 3: Data Loading and Pre-processing Functions
# These functions handle converting the PDF to text and preparing the technology list.

def parse_pdf_to_text(file_path):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text(extraction_mode="layout")
            text += page_text + "\n"
        # Basic cleanup for financial docs
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        return text
    except FileNotFoundError:
        return None

def get_technologies(file_path):
    """Reads Gartner CSV and returns a dictionary of technologies and their variations."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None
    
    synonym_map = {
        "Mac": ["Mac", "macOS", "MacBook"],
        "iPhone": ["iPhone", "iOS"],
        "iPad": ["iPad", "iOS"],
        "Apple Watch": ["Apple Watch", "watchOS"],
        "Apple TV": ["Apple TV", "tvOS"],
        "Apple Pay": ["Apple Pay"],
        "iCloud": ["iCloud", "Cloud Computing"],
    }
    
    tech_list = []
    for techs in df['technologies']:
        tech_list.extend([t.strip() for t in str(techs).split(';')])
    
    for tech in tech_list:
        normalized_tech = re.sub(r'[^\w\s]', '', tech).strip()
        if normalized_tech and normalized_tech not in synonym_map:
            synonym_map[normalized_tech] = [normalized_tech]
            
    return synonym_map
    
# %%
# CELL 4: Context Finding and Analysis Functions
# These functions find the text and perform the sentiment analysis.

def find_context(document, technologies, window=500):
    """Finds mentions and extracts surrounding text using a synonym map."""
    mentions = {}
    for primary_tech, variations in technologies.items():
        for variation in variations:
            pattern = re.compile(r'\b' + re.escape(variation) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(document):
                start = max(0, match.start() - window)
                end = min(len(document), match.end() + window)
                context = document[start:end]
                mentions[primary_tech] = context
                break
            if primary_tech in mentions:
                break
    return mentions

def analyze_with_finbert(text):
    """Analyzes text using the FinBERT model for financial sentiment."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**tokens)
    sentiment_label = model.config.id2label[outputs.logits.argmax().item()]
    return {"Sentiment": sentiment_label}

# %%
# CELL 5: Main Execution Block
# This block ties all the functions together to run the full analysis.

def main():
    print("Step 1: Parsing documents...")
    apple_10q_text = parse_pdf_to_text(TEN_Q_FILE)
    gartner_techs = get_technologies(GARTNER_FILE)

    if apple_10q_text is None or gartner_techs is None:
        print("Error: File not found. Please check your file paths.")
        return

    print("Step 2: Finding technology mentions...")
    tech_mentions = find_context(apple_10q_text, gartner_techs)

    print("Step 3: Analyzing text with FinBERT...")
    if not tech_mentions:
        print("No technologies found in the document.")
        return

    # Create a list to hold the results for the table
    analysis_results = []
    
    for tech, context in tech_mentions.items():
        print(f"  - Analyzing '{tech}'...")
        analysis = analyze_with_finbert(context)
        
        # Append the results to our list
        analysis_results.append({
            "Technology": tech,
            "Sentiment": analysis["Sentiment"]
        })
    
    # 4. Output the results a table
    print("\n--- Final FinBERT Analysis ---")
    print("| Technology | Sentiment |")
    print("|---|---|")
    for result in analysis_results:
        print(f"| {result['Technology']} | {result['Sentiment']} |")
    print("-" * 30)

if __name__ == "__main__":
    main()
# %%
