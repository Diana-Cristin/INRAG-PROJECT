import os
import requests
import gradio as gr

BACKEND = "https://my-back-196080452325.us-central1.run.app"  # cambia si usas otro host/puerto

def subir_pdf(pdf_path: str):
    """Envía el archivo como multipart/form-data con el campo 'file'."""
    if not pdf_path:
        return "Selecciona un PDF primero."
    try:
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            r = requests.post(f"{BACKEND}/upload_pdf", files=files, timeout=120)
        if r.status_code >= 400:
            return f"❌ Error {r.status_code}: {r.text[:800]}"
        data = r.json()
        status = data.get("status", "OK")
        filename = data.get("filename", os.path.basename(pdf_path))
        return f"✅ {status}. Procesado: {filename}"
    except Exception as e:
        # Si la respuesta no es JSON, mostramos el cuerpo de texto
        try:
            return f"❌ Excepción: {e}\nRespuesta: {r.text[:800]}"
        except:
            return f"❌ Excepción: {e}"

def chatear(mensaje: str, historia: list[tuple[str, str]]):
    """Llama a /chat con el query param ?q= ... y agrega la respuesta al Chatbot."""
    if not mensaje:
        return historia
    try:
        # Tu /chat usa Query(...), así que mandamos params, no JSON
        r = requests.post(f"{BACKEND}/chat", params={"q": mensaje}, timeout=120)
        if r.status_code >= 400:
            return historia + [(mensaje, f"❌ Error {r.status_code}: {r.text[:800]}")]
        data = r.json()
        answer = data.get("response", "")
        if not answer:
            return historia + [(mensaje, f"⚠️ Respuesta inesperada: {data}")]
        return historia + [(mensaje, answer)]
    except Exception as e:
        try:
            return historia + [(mensaje, f"❌ Excepción: {e}\nRespuesta: {r.text[:800]}")]
        except:
            return historia + [(mensaje, f"❌ Excepción: {e}")]

with gr.Blocks(title="Chat sobre tu PDF") as demo:
    gr.Markdown("## 📄 Sube tu PDF (campo form-data `file`) y chatea")

    # Subida
    pdf = gr.File(label="PDF", file_types=[".pdf"])
    btn_subir = gr.Button("Subir PDF")
    estado = gr.Textbox(label="Estado", interactive=False)
    btn_subir.click(fn=subir_pdf, inputs=pdf, outputs=estado)

    # Chat
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Pregunta algo sobre el PDF…")
    btn_enviar = gr.Button("Enviar")
    btn_enviar.click(fn=chatear, inputs=[msg, chatbot], outputs=chatbot)

# En Colab puedes usar share=True para obtener un enlace público
port = int(os.getenv("PORT", "8080"))  # Cloud Run asigna el PORT
demo.launch(server_name="0.0.0.0", server_port=port, show_api=False)