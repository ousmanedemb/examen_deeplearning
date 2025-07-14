
# 🗣️ Speech-to-Text + Sentiment Analyzer

Ce projet est une application complète qui permet de :

1. Transcrire automatiquement un **fichier audio (.wav)** en **texte** via un modèle **Wav2Vec2** ;
2. Analyser le **sentiment du texte transcrit** via un modèle de classification fine-tuné (**BERT**) ;
3. Proposer une interface utilisateur simple et interactive avec **Gradio**.

---

## 🧱 Architecture

```
[.wav File] → [Wav2Vec2 Transcription] → [Text]
                                      ↓
                         [BERT Sentiment Classifier]
                                      ↓
                            [Label: Positive/Neutral/Negative]
```

- **Wav2Vec2** (ASR) : pour la transcription vocale.
- **BERT** : pour l’analyse de sentiment.
- **Gradio** : interface web pour interagir en quelques clics.

---

## 🔗 Modèles utilisés

| Tâche                  | Modèle pré-entraîné | Lien Hugging Face |
|------------------------|---------------------|--------------------|
| Transcription vocale   | `facebook/wav2vec2-base-960h` | https://huggingface.co/facebook/wav2vec2-base-960h |
| Sentiment (anglais)    | `bert-base-uncased` fine-tuné | _(modèle entraîné localement)_ |
| Tokenizer              | Basé sur BERT | https://huggingface.co/bert-base-uncased |

Le modèle `sentiment_model/` est entraîné manuellement sur le dataset https://huggingface.co/datasets/SetFit/tweet_sentiment_extraction

---

## 🚀 Lancer l'application

### 1. Installation des dépendances

```bash
git clone https://github.com/ousmanedemb/examen_deeplearning.git
cd examen_deeplearning

python -m venv venv
source venv/bin/activate   # ou venv\Scripts\activate sur Windows

pip install -r requirements.txt
```

> ⚠️ Assurez-vous d'utiliser `numpy<=2.2` pour la compatibilité avec `numba`.

### 2. Lancer l'application
NB : Vous devez executer l'ensemble des cellules du notebook "training_model_bert.ipynb", pour entrainer et sauvegarder le modèle dans le dossier sentiment_model avant de lancer l'application. 
```bash
python app.py
```

Gradio s’ouvrira automatiquement sur http://localhost:7860

---

## 📁 Structure du projet

```
examen_deeplearning/
├── app.py
├── sentiment_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── mapping_labels.json
├── requirements.txt
└── README.md
```

## 💡 Exemples de test

- Upload `.wav` :
    - Positif : "I really love this product!"
    - Négatif : "I'm disappointed, it's not working at all."

---
