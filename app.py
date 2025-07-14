import torch
import torchaudio
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

# 📌 1. Load Wav2Vec2 model for transcription
asr_model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)
asr_model.eval()

# 📌 2. Load sentiment model and tokenizer
sentiment_path = "sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(sentiment_path)
model_load = AutoModelForSequenceClassification.from_pretrained(sentiment_path)
model_load.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_load.to(device)

# 📌 3. Labels
labels = {0: "negative", 1: "neutral", 2: "positive"}


# 🔊 Chargement audio (librosa ou torchaudio)
def load_audio_librosa(path):
    waveform, sr = librosa.load(path, sr=16000, mono=True)
    return torch.tensor(waveform)


# 📝 Transcription
def transcribe(path):
    audio = load_audio_librosa(path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = asr_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.capitalize()


# 🤖 Prédiction de sentiment
def predict_sentiment(text: str):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        output = model_load(**encoding)
        prediction = torch.argmax(output.logits, dim=1).item()
    return labels[prediction]


# 🌐 Gradio interface
def analyze_audio(audio_file):
    transcription = transcribe(audio_file)
    sentiment = predict_sentiment(transcription)
    return transcription, sentiment


interface = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(type="filepath", label="🎤 Téléverse un fichier .wav"),
    outputs=[
        gr.Textbox(label="📝 Transcription"),
        gr.Textbox(label="🧠 Sentiment prédit")
    ],
    title="🗣️ Transcription + Sentiment",
    description="Téléverse un fichier audio pour voir la transcription (Wav2Vec2) et le sentiment associé."
)

if __name__ == "__main__":
    interface.launch()
