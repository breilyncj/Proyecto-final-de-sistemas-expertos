import os
from sentence_transformers import SentenceTransformer
import chromadb

# =============================
#    Conexión a la base de datos vectorial ChromaDB
#    Descripción: Crea un cliente persistente para guardar los embeddings localmente en una carpeta
# =============================
client = chromadb.PersistentClient(path="./chroma_db")


# =============================
#    Crear o recuperar una colección de datos
#    Descripción: Si la colección "naruto_dataset" no existe, la crea; si ya existe, la reutiliza
# =============================
collection = client.get_or_create_collection(name="naruto_dataset")

# =============================
#    Cargar el modelo de embeddings
#    Descripción: Este modelo convierte texto a vectores (usado para búsquedas semánticas)
# =============================
model = SentenceTransformer('all-MiniLM-L6-v2')


# =============================
#    Leer documentos desde una carpeta
#    Descripción: Busca todos los archivos .txt en la carpeta "documentos" y los carga como texto
# =============================
def leer_documentos(ruta_carpeta="documentos"):
    textos = []
    nombres = []
    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith(".txt"):
            ruta = os.path.join(ruta_carpeta, archivo)
            with open(ruta, "r", encoding="utf-8") as f:
                texto = f.read()
                textos.append(texto)
                nombres.append(archivo)
    return textos, nombres


# ======================================================================================
#    Generar y guardar embeddings
#    Descripción: Convierte los textos en vectores (embeddings) y los guarda en ChromaDB
# =======================================================================================
def generar_y_guardar_embeddings():
    textos, nombres = leer_documentos()
    print(f"-> Procesando {len(textos)} documentos para generar embeddings...")

    embeddings = model.encode(textos, show_progress_bar=True)

    for i, emb in enumerate(embeddings):
        # Eliminar condición para asegurarte de que sí los agrega
        collection.add(
            documents=[textos[i]],
            embeddings=[emb.tolist()],
            ids=[nombres[i].replace(" ", "_").replace(".txt", "")]

        )
    
    print("Embeddings generados y guardados correctamente.")


# ========================================================================================
#    Ejecución del script
#    Descripción: Solo se ejecuta si corres este archivo directamente
# =========================================================================================
if __name__ == "__main__":
    generar_y_guardar_embeddings()

