#%%
# Imports
#---------

import pandas as pd
import re
import os
from pypdf import PdfReader
from openai import OpenAI
import json
from dotenv import load_dotenv

# %% 
# Load environment variables from the .env file
load_dotenv()

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define file paths using relative paths that are easy to manage
TEN_Q_FILE = "data/10Q (2017-02-01) AAPL.pdf"
GARTNER_FILE = "data/Gartner_Hype_Cycle__publicly_listed_items__2000_2025.csv"

# %% 
# Extracting text from PDF File
#---------------------------------

def parse_pdf_to_text(file_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text(extraction_mode="layout") # The extraction preserves original layout of the page
        text += page_text + "\n" # This adds content from different pages separately
    # This cleans up broken text from hyphens/line breaks and joins them together (r'\1\2')
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text) 
    return text

# Extracting technologies from the CSV file
#------------------------------------------

def get_technologies(file_path):
    """Reads Gartner CSV and returns a dictionary of technologies and their variations."""
    df = pd.read_csv(file_path)
    
    # Dictionary of hard-coded list from the PDF.
    synonym_map = {
        "Mac": ["Mac", "macOS", "MacBook"],
        "iPhone": ["iPhone", "iOS"],
        "iPad": ["iPad", "iOS"],
        "Apple Watch": ["Apple Watch", "watchOS"],
        "Apple TV": ["Apple TV", "tvOS"],
        "Apple Pay": ["Apple Pay"],
        "iCloud": ["iCloud", "Cloud Computing"],
        # Add more synonyms as you find them in the document
    }

    # Adding the technologies from CSV to a list.
    tech_list = []
    for techs in df['technologies']:
        tech_list.extend([t.strip() for t in str(techs).split(';')])

    print("\n--- Initial technology list from CSV (before normalization) ---")
    print(tech_list)
    
    # Normalization
    for tech in tech_list:
        normalized_tech = re.sub(r'[^\w\s]', '', tech).strip() # removes non-word characters such as ;
        if normalized_tech and normalized_tech not in synonym_map:
            synonym_map[normalized_tech] = [normalized_tech] # adds new tech to synonym map
            
    print("\n--- Final technology synonym map ---")
    print(synonym_map)
    
    return synonym_map

# %%
# Finding mentioned techs in PDF
#---------------------

def find_context(document, technologies, window=500):
    """Finds mentions and extracts surrounding text using a synonym map."""
    mentions = {}
    for primary_tech, variations in technologies.items():
        # Iterate through all known variations for a technology
        for variation in variations:
            # Build a flexible regex pattern for matching
            pattern = re.compile(r'\b' + re.escape(variation) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(document):
                start = max(0, match.start() - window)
                end = min(len(document), match.end() + window)
                context = document[start:end]
                mentions[primary_tech] = context
                # Stop after the first mention for simplicity
                break
            if primary_tech in mentions:
                break # Break out of the inner loop if a match is found
    
    # See the contexts found
    print("\n--- Contexts found in the document ---")
    for tech, context in mentions.items():
        print(f"Technology: {tech}")
        print(f"Context: {context[:100]}...") # Print first 100 chars for brevity
        print("-" * 20)
    
    return mentions

# %%
def analyze_with_openai(text, technology):
    """Sends text to OpenAI API for detailed analysis."""
    prompt = f"""
    Analyze the following text from a financial report related to "{technology}".
    The text is: "{text}"
    Provide a JSON object with:
    "Sentiment": ("Positive", "Negative", or "Neutral"),
    "ToneAndStyle": ("Formal", "Cautious", "Informative"),
    "Emotion": ("Confident", "Concern", "Neutral"),
    "Intent": "Describe the purpose (e.g., risk disclosure, performance summary).",
    "Stance": ("Agreement", "Contradiction", "Neutrality").
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def main():
    # 1. Parse and extract data
    print("Step 1: Parsing documents...")
    try:
        apple_10q_text = parse_pdf_to_text(TEN_Q_FILE)
        gartner_techs = get_technologies(GARTNER_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure your file paths are correct.")
        return

    # 2. Find and extract context
    print("Step 2: Finding technology mentions...")
    tech_mentions = find_context(apple_10q_text, gartner_techs)

    # 3. Analyze each mention
    print("Step 3: Analyzing text with OpenAI API...")
    if not tech_mentions:
        print("No technologies found in the document.")
        return

    for tech, context in tech_mentions.items():
        print(f"  - Analyzing '{tech}'...")
        analysis = analyze_with_openai(context, tech)

        # 4. Output the results
        print("\n--- Final Analysis for:", tech, "---")
        print(json.dumps(analysis, indent=2))
        print("-" * 30)

if __name__ == "__main__":
    main()

# %%
