# verificar_index.py
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="naruto_dataset")

print("📂 Archivos indexados en la colección 'naruto_dataset':\n")
docs = collection.get()
for i, doc in enumerate(docs['documents']):
    print(f"{i+1}. ID: {docs['ids'][i]} - {doc[:100]}...")  # Muestra solo primeros 100 caracteres
