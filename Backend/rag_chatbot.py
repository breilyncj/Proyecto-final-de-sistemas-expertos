import os
import re
import chromadb
from dotenv import load_dotenv   
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# =============================
#    Cargar claves de entorno
# =============================
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# =============================
#     Modelo de Embeddings
# =============================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# =============================
#  Configuración de ChromaDB
# =============================
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="naruto_dataset")

# =============================
#     Funciones auxiliares
# =============================
def generar_embedding_pregunta(pregunta):
    return embedding_model.encode([pregunta])[0].tolist()

def limpiar_markdown(texto):
    """Elimina caracteres especiales usados en formato Markdown."""
    return re.sub(r'[*_`]', '', texto)

def buscar_contexto(pregunta_embedding, k=10):
    resultados = collection.query(
        query_embeddings=[pregunta_embedding],
        n_results=k,
        include=['documents']
    )
    documentos_encontrados = resultados['documents'][0]
    return "\n\n".join(documentos_encontrados)

def generar_respuesta(pregunta, contexto):
    prompt = f"""
Eres un asistente experto en Naruto. Responde sin frases tipo "según el texto proporcionado". Responde preguntas basándote en el siguiente contexto:

{contexto}

Pregunta: {pregunta}
"""
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


# =============================
#      Función principal
# =============================
def main():
    print("🤖 NarutoBot con Gemini (escribe 'salir' para terminar)\n")
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() in ["salir", "Adiós", "adiós", "adios", "Adios", "chao", "Chao", "exit", "quit"]:
            print("Hasta luego 🍥")
            break

        print(" Buscando información...")
        embedding = generar_embedding_pregunta(pregunta)
        contexto = buscar_contexto(embedding)

        print("...Generando respuesta...\n")
        respuesta = generar_respuesta(pregunta, contexto)
        print("NarutoBot:", respuesta)
        print("-" * 50)


# =============================
#      Ejecutar script
# =============================
if __name__ == "__main__":
    main()