import io
import os
import datetime
import numpy as np
import gradio as gr
from PIL import Image
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

def call_backend(image: Image.Image):
    """Envoie l'image au backend FastAPI (Membre 3 + modèle de Membre 2) et retourne
    (label_fr, score, is_real)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    response = requests.post(
        f"{API_URL}/predict",
        files={"file": ("image.jpg", buf, "image/jpeg")},
        timeout=120,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"Erreur API ({response.status_code}) : {detail}")
    data = response.json()

    # TODO: vérifiez auprès de Membre 3 les valeurs EXACTES renvoyées par son
    # schema PredictionResponse (champ "label") — ex. "real" / "manipulated",
    # et adaptez ce mapping en conséquence.
    is_real = data["label"].lower() in ("real", "réelle", "reel")
    label_fr = "Image réelle" if is_real else "Image générée par une IA"
    return label_fr, float(data["confidence"]), is_real

def analyze_image(image_path):
    if image_path is None:
        return ("—", None, "", None)
    img = Image.open(image_path).convert("RGB")
    
    try:
        label, score, is_real = call_backend(img)
    except requests.exceptions.RequestException as e:
        error_html = (
            '<div style="color:#dc2626;background:#fef2f2;border:1px solid #fecaca;'
            f'padding:15px;border-radius:10px;">Impossible de contacter le serveur '
            f'd\'analyse. Vérifiez que l\'API est démarrée. ({e})</div>'
        )
        return (error_html, img, "Erreur de connexion", None)
    
    pct = round(score * 100, 1)
    if is_real:
        color = "#15803d"
        bg = "#f0fdf4"
        border = "#bbf7d0"
    else:
        color = "#dc2626"
        bg = "#fef2f2"
        border = "#fecaca"
    if pct >= 90:
        confidence_word = "Niveau de confiance : très élevé"
    elif pct >= 75:
        confidence_word = "Niveau de confiance : élevé"
    else:
        confidence_word = "Niveau de confiance : moyen"
    result_html = f"""
    <div style="background:{bg};border:1px solid {border};border-radius:16px;padding:26px;text-align:center;margin-top:10px;">
        <div style="color:#64748b;font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Résultat de l'analyse</div>
        <div style="color:{color};font-size:25px;font-weight:750;margin-bottom:6px;">{label}</div>
        <div style="color:{color};font-size:38px;font-weight:800;margin:5px 0 12px 0;">{pct} %</div>
        <div style="width:100%;background:#e2e8f0;border-radius:10px;height:9px;overflow:hidden;margin:12px 0;">
            <div style="width:{pct}%;background:{color};height:100%;border-radius:10px;"></div>
        </div>
        <div style="color:#64748b;font-size:13px;margin-top:10px;">{confidence_word}</div>
        <div style="color:#94a3b8;font-size:11px;margin-top:5px;">Analyse effectuée localement</div>
    </div>
    """
    return (result_html, img, f"{label} — {pct} %", label)

CSS = """
body { background: #f8fafc !important; }
#hero { text-align: center; padding: 65px 25px 40px 25px; }
#hero h1 { font-size: 44px; font-weight: 800; color: #0f172a; margin-bottom: 12px; }
#hero p { font-size: 17px; line-height: 1.7; color: #64748b; max-width: 700px; margin: 0 auto 28px auto; }

.start-btn-container { max-width: 280px; margin: 0 auto; }

button.gr-button {
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    min-height: 48px !important;
}

.card { border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; background: white; min-height: 150px; box-shadow: 0 4px 15px rgba(15, 23, 42, 0.04); }
.card h3 { color: #0f172a; font-size: 18px; margin-bottom: 8px; }
.card p { color: #64748b; line-height: 1.6; }

#detector-title { font-size: 28px; font-weight: 800; color: #0f172a; margin-bottom: 5px; }
#detector-sub { color: #64748b; font-size: 14px; line-height: 1.6; margin-bottom: 18px; }

.back-btn-container { max-width: 230px; margin-bottom: 20px; }

.reset-btn button {
    background: white !important;
    color: #475569 !important;
    border: 1px solid #cbd5e1 !important;
}

.analyze-btn button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
}
.analyze-btn button:hover { background: #1d4ed8 !important; }

.gradio-container { max-width: 1200px !important; margin: auto !important; }
"""

with gr.Blocks(css=CSS, title="Détection d'images par IA") as demo:
    history_state = gr.State([])
    full_records_state = gr.State([])
    
    with gr.Column(visible=True) as page_landing:
        with gr.Column(elem_id="hero"):
            gr.HTML("<h1>Détection d'images par IA</h1>")
            gr.HTML("<p>Analysez une image afin d'estimer si elle correspond à une photographie réelle ou à une image générée par une intelligence artificielle.</p>")
            with gr.Row(elem_classes=["start-btn-container"]):
                start_btn = gr.Button("Accéder au détecteur", variant="primary", size="lg")
        with gr.Row():
            gr.HTML('<div class="card" style="margin:10px;"><h3>Analyse rapide</h3><p>Obtenez une estimation en quelques secondes directement depuis l\'application.</p></div>')
            gr.HTML('<div class="card" style="margin:10px;"><h3>Résultat détaillé</h3><p>Consultez la classification obtenue ainsi que le niveau de confiance associé.</p></div>')
            gr.HTML('<div class="card" style="margin:10px;"><h3>Historique interactif</h3><p>Retrouvez et consultez les détails de vos analyses précédentes à tout moment.</p></div>')

    with gr.Column(visible=False) as page_detector:
        with gr.Row(elem_classes=["back-btn-container"]):
            back_btn = gr.Button("← Retour à l'accueil", elem_classes=["reset-btn"])
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div id="detector-title">Analyse de l\'image</div><div id="detector-sub">Sélectionnez une image afin de lancer la détection.</div>')
                image_input = gr.Image(type="filepath", label="Sélectionner une image", sources=["upload"])
                with gr.Row():
                    cancel_btn = gr.Button("Annuler", elem_classes=["reset-btn"])
                    new_img_btn = gr.Button("Nouvelle analyse", elem_classes=["reset-btn"])
            with gr.Column(scale=1):
                preview = gr.Image(label="Aperçu de l'image", interactive=False)
                with gr.Row():
                    analyze_btn = gr.Button("Analyser l'image", elem_classes=["analyze-btn"])
                result_box = gr.HTML()
        
        with gr.Column():
            gr.HTML('<div style="font-weight:700;margin-top:25px;margin-bottom:8px;color:#1e293b;font-size:18px;">Historique des analyses (Cliquez sur une ligne pour voir les détails)</div>')
            history_table = gr.Dataframe(
                headers=["Heure", "Résultat", "Nom du fichier"],
                datatype=["str", "str", "str"],
                value=[],
                interactive=False,
                row_count=(1, "dynamic"),
                col_count=(3, "fixed")
            )

    def go_to_detector():
        return (gr.update(visible=False), gr.update(visible=True))

    def go_to_landing():
        return (gr.update(visible=True), gr.update(visible=False))

    start_btn.click(go_to_detector, outputs=[page_landing, page_detector])
    back_btn.click(go_to_landing, outputs=[page_landing, page_detector])

    def on_analyze(image_path, history_data, full_records):
        if image_path is None:
            return ('<div style="color:#dc2626;background:#fef2f2;border:1px solid #fecaca;padding:15px;border-radius:10px;">Veuillez sélectionner une image avant de lancer l\'analyse.</div>', None, history_data, full_records)
        
        result_html, prev_img, summary, label = analyze_image(image_path)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        
        record = {
            "time": ts,
            "summary": summary,
            "image": image_path,
            "html": result_html
        }
        full_records = [record] + full_records
        
        # Utilisation de os.path.basename pour ne garder que le nom du fichier (ex: "photo.jpg")
        table_values = [[r["time"], r["summary"], os.path.basename(r["image"])] for r in full_records]
        
        return (result_html, prev_img, table_values, full_records)

    analyze_btn.click(
        on_analyze, 
        inputs=[image_input, history_table, full_records_state], 
        outputs=[result_box, preview, history_table, full_records_state]
    )

    def select_history_item(evt: gr.SelectData, full_records):
        if full_records and 0 <= evt.index[0] < len(full_records):
            selected = full_records[evt.index[0]]
            return selected["html"], selected["image"]
        return gr.update(), gr.update()

    history_table.select(
        select_history_item,
        inputs=[full_records_state],
        outputs=[result_box, preview]
    )

    image_input.change(lambda p: p, inputs=[image_input], outputs=[preview])

    def on_reset():
        return (None, None, "")

    cancel_btn.click(on_reset, outputs=[image_input, preview, result_box])
    new_img_btn.click(on_reset, outputs=[image_input, preview, result_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8000)))