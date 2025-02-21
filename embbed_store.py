import os
import sqlite3
from sentence_transformers import SentenceTransformer
import markdown
import numpy as np
import pickle

# SentenceTransformer model for embedding text
model = SentenceTransformer('all-MiniLM-L6-v2')

# paths
md_file_path = 'D:/MRI_Scan_Prediction/deploy/Tumor_description.md'
db_path = 'D:/MRI_Scan_Prediction/deploy/tumor_info.db'

# To read md file
def read_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

md_content = read_markdown(md_file_path)

# md to text
def markdown_to_text(md_content):
    html_content = markdown.markdown(md_content)
    return html_content

text_content = markdown_to_text(md_content)

# Split text into chunks 
paragraphs = text_content.split('\n')

# Embed the text chunks 
embeddings = model.encode(paragraphs)

# database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Table for storing tumor information
cursor.execute('''
    CREATE TABLE IF NOT EXISTS tumor_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT,
        embedding BLOB
    )
''')

# Inserting into database
for paragraph, embedding in zip(paragraphs, embeddings):
    cursor.execute('''
        INSERT INTO tumor_info (description, embedding) VALUES (?, ?)
    ''', (paragraph, pickle.dumps(embedding)))

conn.commit()
conn.close()

print(f"Embeddings saved to SQLite database at {db_path}")