# app_standalone.py
import gradio as gr
import torch
import librosa
import numpy as np
import soundfile as sf
import io
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import os
# --- Model Loading Logic (from the old main.py) ---
MODEL_PATH = "./best_model"
CONFIDENCE_THRESHOLD = 0.50 # 50%

print("Loading model...")
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    model = ASTForAudioClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # In a real app, you might want to exit or handle this more gracefully
    raise RuntimeError(f"Could not load model from {MODEL_PATH}") from e


# --- Inference Function ---
def classify_audio(audio_input):
    if audio_input is None: return {"No audio provided": 0.0}
    sr, audio_data = audio_input

    # Normalize audio
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
    
    # Preprocess and predict
    inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    scores = torch.nn.functional.softmax(logits, dim=-1)
    top_k_scores, top_k_indices = torch.topk(scores, 5, dim=-1)
    
    results = []
    for i in range(5):
        confidence = top_k_scores[0][i].item()
        class_id = top_k_indices[0][i].item()
        label = model.config.id2label[class_id]
        results.append({"sound_class": label, "confidence": confidence})

    # Filter by confidence and format for Gradio
    top_prediction_confidence = results[0]['confidence']
    if top_prediction_confidence < CONFIDENCE_THRESHOLD:
        return {"Uncertain / Background Noise": top_prediction_confidence}
    
    return {result['sound_class']: result['confidence'] for result in results}

# --- Gradio UI (from the old app.py) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # AcousticSense: Urban Sound Classification (All-in-One)
        ## Use your microphone to record a short clip or upload an audio file. The model runs directly in this service.
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("Record from Microphone"):
            mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Record a 3-4 second audio clip")
            record_button = gr.Button("Classify Recording", variant="primary")
            
        with gr.TabItem("Upload Audio File"):
            file_input = gr.Audio(sources=["upload"], type="numpy", label="Upload an audio file (.wav, .mp3, etc.)")
            upload_button = gr.Button("Classify Uploaded File", variant="primary")

    gr.Markdown("### Predictions")
    output_label = gr.Label(num_top_classes=5, label="Top 5 Sound Classes")

    record_button.click(fn=classify_audio, inputs=mic_input, outputs=output_label)
    upload_button.click(fn=classify_audio, inputs=file_input, outputs=output_label)

# --- Launch Command for Cloud Run ---
if __name__ == "__main__":
    # The server_name="0.0.0.0" is crucial for Docker
    # Most cloud services provide the port they expect the app to listen on via the PORT environment variable.
    # We'll use that, and default to 8080 if it's not set.
    server_port = int(os.environ.get('PORT', 8080))
    demo.launch(server_name="0.0.0.0", server_port=server_port)
