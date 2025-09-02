# Imports necesarios
import os
from dotenv import load_dotenv
load_dotenv()
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import easyocr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from fastapi import FastAPI, UploadFile, Query, HTTPException
import uvicorn
from pyngrok import ngrok
import nest_asyncio
from typing import List

# Configurar Tesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Inicializar EasyOCR
reader = easyocr.Reader(['es'])

# Inicializar modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = []
index = None


# OCR
def extract_text_with_tesseract(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        for img in page.get_images():
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            text += pytesseract.image_to_string(image, lang='spa')
    return text

def extract_text_with_easyocr(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        for img in page.get_images():
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            result = reader.readtext(image)
            text += " ".join([detection[1] for detection in result])
    return text

#RAG
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List

model = SentenceTransformer('all-MiniLM-L6-v2')
documents = []  # Aquí guardaremos los chunks de texto
index = None

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Divide el texto en chunks de un tamaño específico."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_index(texts: List[str]):
    global documents, index
    # Dividir cada texto en chunks
    all_chunks = []
    for text in texts:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    documents = all_chunks

    # Generar embeddings para cada chunk
    embeddings = model.encode(documents, convert_to_tensor=True)

    # Crear el índice FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.cpu().numpy())
    return index

def search_index(query: str, k: int = 3) -> List[str]:
    """Busca los chunks más relevantes para la consulta."""
    query_embedding = model.encode([query], convert_to_tensor=True)
    distances, indices = index.search(query_embedding.cpu().numpy(), k)
    return [documents[i] for i in indices[0]]

#GROQ
def generate_response(prompt, context):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Lee la clave desde variable de entorno
    messages = [
        {"role": "system", "content": f"Contexto: {context}"},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.5
    )
    return response.choices[0].message.content

# FASTAPI
app = FastAPI()

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile):

    # Usar una ruta relativa y crear la carpeta si no existe
    content_dir = "./content"
    os.makedirs(content_dir, exist_ok=True)
    pdf_path = os.path.join(content_dir, file.filename)
    with open(pdf_path, "wb") as f:
        f.write(file.file.read())

    # Usa Tesseract o EasyOCR según necesites
    text = extract_text_with_tesseract(pdf_path)
    # text = extract_text_with_easyocr(pdf_path)

    create_index([text])
    return {"status": "PDF procesado", "filename": file.filename}

@app.post("/chat")
async def chat(q: str = Query(...)):
    if not index:
        raise HTTPException(status_code=400, detail="Sube un PDF primero")
    context = "\n".join(search_index(q))
    response = generate_response(q, context)
    return {"response": response}




#------------
from fastapi import FastAPI
import nest_asyncio
from pyngrok import ngrok
import uvicorn

# Permitir anidamiento de loops de eventos
nest_asyncio.apply()
print("iniciando server")


# Iniciar FastAPI
uvicorn.run(app, port=8080)