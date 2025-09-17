# tools.py
import os
import re
import requests
from dotenv import load_dotenv
from groq import Groq
from PyPDF2 import PdfReader
import sympy as sp
import numpy as np

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
news_api_key = os.getenv("NEWS_API_KEY")




from sympy import sympify
import re# ---- Calculator ----
import re
from sympy import sympify

math_keywords = ["add", "subtract", "plus", "minus", "times", "multiply", "divide", "by", "percentage", "sqrt"]

def is_simple_math(user_input):
    return re.fullmatch(r"[0-9\+\-\*\/\^\(\)\. ]+", user_input.strip()) is not None

import re
from sympy import sympify

def parse_math_expression(text):
    text = text.lower().strip()

    # Replace words with operators
    replacements = {
        r'\badd\b': '+', 
        r'\bplus\b': '+',
        r'\bsubtract\b': '-', 
        r'\bminus\b': '-',
        r'\bmultiply\b': '*', 
        r'\btimes\b': '*', 
        r'\bx\b': '*',
        r'\bdivide\b': '/', 
        r'\bby\b': '/',
        r'\band\b': '+' 
    }
    for k, v in replacements.items():
        text = re.sub(k, v, text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^\+|\+$', '', text).strip()  

    tokens = re.findall(r'\d+\.?\d*|[+\-*/()]', text)
    if not tokens:
        return None

    expr = ''
    for i, token in enumerate(tokens):
        expr += token
       
        if i < len(tokens) - 1 and token.replace('.', '', 1).isdigit() and tokens[i + 1].replace('.', '', 1).isdigit():
            expr += '+'
    if not expr or re.match(r'^[+\-*/]', expr) or re.search(r'[+\-*/]{2,}', expr):
        return None

    return expr

def calculate(user_input):
    expr = parse_math_expression(user_input)
    if expr:
        try:
            return sympify(expr).evalf()
        except Exception as e:
            return f"Error: {e}"
    else:
        return "Could not parse expression."





# ---------------- PDF ----------------
def summarize_pdf(file_path, chunk_size=1000):
    try:
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''

        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = []
        for chunk in chunks:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {'role': 'system', 'content': 'You are a helpful AI assistant.'},
                    {'role': 'user', 'content': f'Summarize this text:\n{chunk}'}
                ]
            )
            summaries.append(response.choices[0].message.content)
        return " ".join(summaries)
    except Exception as e:
        return f'Error reading the pdf: {str(e)}'


# ---------------- News ----------------
# In fetch_news: add timeout and handle missing API key
def fetch_news(query, page_size=10):
    if not news_api_key:
        return "News API key not configured."
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&pageSize={page_size}&apiKey={news_api_key}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("status") != "ok":
            return f"Error fetching news: {data.get('message', 'Unknown error')}"
        articles = data.get("articles", [])
        if not articles:
            return "No recent news found."
        news_text = ""
        for i, article in enumerate(articles, 1):
            news_text += f"{i}. {article['title']} ({article['source']['name']})\n{article['url']}\n\n"
        return news_text
    except Exception as e:
        return f"Error fetching news: {e}"
