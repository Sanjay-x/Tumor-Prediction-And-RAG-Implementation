import cohere
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import pickle
from sentence_transformers import SentenceTransformer

# Cohere API
cohere_client = cohere.Client('API KEY')  

# Load model
model = load_model('fine_tuned_model.keras')

# model for embedding text
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# database setup
db_path = 'D:/MRI_Scan_Prediction/deploy/tumor_info.db'

# mapping
class_mapping = {
    0: "Glioma",
    1: "Meningioma",
    2: "No tumor",
    3: "Pituitary"
}

# classification function
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    tumor_type = class_mapping.get(predicted_class, "Unknown")

    return tumor_type

# To retrieve information 
def retrieve_information(query_text):
    
    query_embedding = sentence_model.encode(query_text)  
    
    # Connect to database
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    # Fetch embeddings and descriptions from database
    cursor.execute("SELECT description, embedding FROM tumor_info")
    rows = cursor.fetchall()

    embeddings = [pickle.loads(row[1]) for row in rows] 
    documents = [row[0] for row in rows]  

    # similarities between the query and stored embeddings
    similarities = cosine_similarity([query_embedding], embeddings)
    
    # similar document
    most_similar_idx = np.argmax(similarities)
    response = documents[most_similar_idx]

    # format the response
    cleaned_response = clean_html(response)

    # prompt
    enhanced_prompt = (
      f"You are a highly skilled medical assistant. Given the query '{query_text}', provide a highly detailed, structured, empathetic response covering the following sections:\n\n"
      f"1. **Symptoms**: Describe symptoms in great depth, starting with the earliest signs and how they manifest during the initial stages. "
      f"Then, explain how symptoms evolve and progress over time, mentioning any rare or atypical signs that may appear in advanced stages.\n"
      f"2. **Actionable Insights**: Provide in-depth, practical advice for recognizing early symptoms, including guidance on when to seek medical attention and how to manage symptoms early on.\n"
      f"3. **Treatment Considerations**: Elaborate on how symptoms influence treatment choices and why timely intervention is critical.\n"
      f"4. **Prognosis and Management**: Provide a clear overview of the prognosis and management strategies for this type of tumor, emphasizing long-term outcomes and potential challenges.\n\n"
      f"Response based on the extracted data: {cleaned_response}\n"
      f"Please ensure the response is long, highly informative, empathetic, and well-structured. Prioritize clarity and real-world application of medical advice."
    )

    # Generate using Cohere 
    cohere_response = cohere_client.generate(
        model='command-xlarge', 
        prompt=enhanced_prompt,
        max_tokens=20000, 
        temperature=0.7,  
        stop_sequences=["\n"]
    )

    conn.close()

    return cohere_response.generations[0].text.strip()

# Function to clean the retrieved content
def clean_html(text):
    import re
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)

# Example 
if __name__ == "__main__":
    img_path = "D:/MRI_Scan_Prediction/Testing/meningioma/Te-me_0022.jpg"  
    tumor_type = classify_image(img_path)
    print(f"Predicted tumor type: {tumor_type}")
    query_text = f"Provide a detailed medical and clinical overview of {tumor_type}, treatment options, prognosis, and relevant case studies or research findings."
    info = retrieve_information(query_text)
    print(f"Retrieved information: {info}")
