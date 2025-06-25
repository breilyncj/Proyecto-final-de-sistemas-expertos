from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_chatbot import generar_embedding_pregunta, buscar_contexto, generar_respuesta

# =============================
#    Crear instancia FastAPI
# =============================
app = FastAPI()

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
#      Modelo para petición
# =============================
class Pregunta(BaseModel):
    pregunta: str

# =============================
#         Endpoint API
# =============================
@app.post("/preguntar")
async def preguntar(data: Pregunta):
    embedding = generar_embedding_pregunta(data.pregunta)
    contexto = buscar_contexto(embedding)
    respuesta = generar_respuesta(data.pregunta, contexto)
    return {"respuesta": respuesta}
