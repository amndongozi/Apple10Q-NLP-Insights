*_**FinBERT & OpenAI: Apple 10-Q Technology Sentiment Analysis***_

This project is a Natural Language Processing (NLP) tool built to automatically extract and analyze technology mentions from SEC financial filings. By using a combination of specialized models and data analysis libraries, it provides a straightforward and effective way to gain insights from complex financial documents.

***Tool Stack***

**Python**: Core programming language.

**Git**: Version control.

**Hugging Face transformers**: For FinBERT model.

**FinBERT**: Specialized financial sentiment model.

**OpenAI API**: For nuanced sentiment analysis.

**pandas**: For data handling.

**uv or pip**: Package management.

***How It Works***

1. Read Files: The script reads the Apple 10-Q and Gartner technology list.

2. Find Context: It finds and extracts surrounding text for each technology mention.

3. Analyze Sentiment: Both FinBERT and OpenAI analyze each text snippet for sentiment.

***How to Get Started***

To run this project, you need a Python environment and the required libraries.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/amndongozi/Apple10Q-NLP-Insights.git
    cd Apple10Q-NLP-Insights
    ```
2.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    ```bash
    python main.py
    ```

