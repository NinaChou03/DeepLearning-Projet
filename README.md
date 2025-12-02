# DeepLearning-Projet :
# Multimodal Sentiment Analysis — MELD & MOSI

Deep Learning Project — Multimodal Emotion/Sentiment Understanding

Ce dépôt contient deux expériences complètes d’analyse multimodale des émotions et du sentiment :

MELD : Modèle multimodal basé sur BERT + Audio + Vision Symbolique (DEVA-like)

MOSI : Modèle pour sentiment analysis sur données audio-vidéo avec features text/audio/visuel

Comparaison des performances

## Structure du Dépôt

multimodal-sentiment-analysis
│

├── MELD_model.ipynb         # Code complet du modèle pour MELD

├── MOSI_model.ipynb         # Code complet du modèle pour MOSI

│

├── models/                  # (optionnel) sauvegardes des modèles

├── data/                    # téléchargée automatiquement (kagglehub)

│

├── Deep Learning - prez projet.pptx   # Présentation du projet

└── README.md

## Objectif du Projet

Le but est d'explorer comment combiner Texte + Audio + Vision pour améliorer la prédiction du sentiment et des émotions humaines.

✔ Analyse multimodale

✔ Extraction automatique de caractéristiques audio (pitch, loudness, MFCC…)

✔ Modèle Transformer pour fusion inter-modale

✔ Comparaison sur deux datasets majeurs : MELD et CMU-MOSI


## Technologies :

Python 3.10+ / PyTorch / Transformers (HuggingFace) / Librosa / scikit-learn / KaggleHub / NumPy / Pandas /Matplotlib

## Datasets Utilisés

1️⃣ MELD — Multimodal EmotionLines Dataset

Contient des dialogues extraits de la série Friends, annotés avec :

Texte + Audio + Vidéo

Sentiment : négatif / neutre / positif

Émotion : joie, colère, tristesse, surprise, dégoût…

Chargement automatique dans le code :


import kagglehub

path = kagglehub.dataset_download("zaber666/meld-dataset")

2️⃣ CMU-MOSI

Dataset d’analyse de sentiment multimodal :

Vidéos YouTube de monologues

Transcriptions

Audio (pitch, MFCC…)

Sentiment entre −3 et +3

## Modèle MELD – Architecture Résumée

Le modèle implémente une approche inspirée de DEVA :

#### 1. Encodage Texte

Tokenisation avec BERT / DistilBERT

Passage dans un TransformerEncoder

Extraction des embeddings séquentiels

#### 2. Encodage Audio

Extraction MFCC

Pitch / Loudness / Jitter / Shimmer transformés en texte

Embedding via BERT → fusion avec features MFCC

#### 3. Vision Symbolique (VED)

Transforme l’émotion annotée en phrases décrivant les AUs :

Exemple :
“The person shows signs of raised cheek and pulled lip corner.”

#### 4. Fusion Modale
Basée sur Cross-Modal Attention (CMA) et MFU (Modal Fusion Units)
permettant au texte de s’ajuster grâce à l’audio et la vision.

#### 5. Régression du Sentiment
Sortie ∈ [-1, +1]

## Métriques Calculées
Les métriques demandées dans le projet sont implémentées :

Metric	Description
ACC-2	Classification binaire (sent négatif vs positif/neutre)
ACC-5	Version 5 classes (binning)
F1 Score	F1 weighted
MAE	Mean Absolute Error (régression)
Corr	Corrélation de Pearson entre prédiction et vérité

## Résultats Résumés
✔ MELD (DEVA-like Model)
Metric	Score
ACC-2	0.94
ACC-3 0.93
ACC-5 0.86
ACC-7 0.86
F1	0.94
MAE	0.26
Corr	0.85
Train loss 0.17

✔ MOSI (simple multimodal fusion)
Metric	Score
ACC-2	0.86
ACC-5 0.56
ACC-7 0.76
F1	0.85
MAE	0.47
Corr	0.95


## Exécution
1. Installer les dépendances :

pip install torch transformers librosa kagglehub scikit-learn matplotlib

3. Lancer les notebooks :

lancer dans google collab :

MOSI_Deeplearning.ipynb 
MELD_projet.ipynb


Tout se charge automatiquement.

## Conclusion du Projet

L’audio apporte des indices subtils (pitch, jitter…) utiles au sentiment.

Le texte reste la modalité la plus puissante.

La fusion multimodale par Transformers + Attention améliore les performances.

MELD est plus difficile que MOSI car :

dialogues plus courts

dataset plus bruité

pus de classes émotionnelles

## Auteurs
Nina Tuil - Arthur Neyt - Loic Le Quellec
Master IA & Data — 2025-2026
