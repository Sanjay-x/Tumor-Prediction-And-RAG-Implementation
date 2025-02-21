import chromadb

# Chroma client
client = chromadb.Client()

# Chroma collection 
collection = client.create_collection("tumor_info")

