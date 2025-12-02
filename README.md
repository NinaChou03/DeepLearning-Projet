# DeepLearning-Projet :
# Multimodal Sentiment Analysis — MELD & MOSI

Deep Learning Project — Multimodal Emotion/Sentiment Understanding

Ce dépôt contient deux expériences complètes d’analyse multimodale des émotions et du sentiment :

MELD : Modèle multimodal basé sur BERT + Audio + Vision Symbolique (DEVA-like)

MOSI : Modèle pour sentiment analysis sur données audio-vidéo avec features text/audio/visuel

Comparaison des performances

## Structure du Dépôt

```
multimodal-sentiment-analysis/
│
├── MELD_model.ipynb             # Notebook complet du modèle MELD
├── MOSI_model.ipynb             # Notebook complet du modèle MOSI
│
├── models/                      # Sauvegardes .pt
├── data/                        # Données téléchargées automatiquement via kagglehub
│
├── requirements.txt             # Dépendances
├── Deep Learning - prez projet.pptx
└── README.md
```

## Objectif du Projet

Le but est d'explorer comment combiner Texte + Audio + Vision pour améliorer la prédiction du sentiment et des émotions humaines.

✔ Analyse multimodale

✔ Extraction automatique de caractéristiques audio (pitch, loudness, MFCC…)

✔ Modèle Transformer pour fusion inter-modale

✔ Comparaison sur deux datasets majeurs : MELD et CMU-MOSI

## Installation : 

```
git clone https://github.com/<username>/multimodal-sentiment-analysis.git
cd multimodal-sentiment-analysis

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Technologies :

Python 3.10+ / PyTorch / Transformers (HuggingFace) / Librosa / scikit-learn / KaggleHub / NumPy / Pandas /Matplotlib

## Datasets Utilisés

1️⃣ MELD — Multimodal EmotionLines Dataset

Contient des dialogues extraits de la série Friends, annotés avec :

Texte + Audio + Vidéo

Sentiment : négatif / neutre / positif

Émotion : joie, colère, tristesse, surprise, dégoût…

Chargement automatique dans le code :

```
import kagglehub
path = kagglehub.dataset_download("zaber666/meld-dataset")
```

Cependant pour des moyens pratique nous avons opter pour telecharger le dossier complet de manière à le manipuler plus facilement. 

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

## Notre pipeline pour MELD : 

### 1. Préparation des données

1. Chargement des CSV : utterance, émotion, sentiment, IDs.
2. Reconstruction des chemins audio/vidéo.
3. Gestion des fichiers audio manquants ou illisibles via fallback neutre.
4. Encapsulation dans un Dataset PyTorch + DataLoader.

### 2. Encodage Texte (BERT + CEU)

1. Tokenisation (BertTokenizerFast).
2. Passage dans BERT pour obtenir les embeddings.
3. Ajout d’un modality token.
4. Passage dans un TransformerEncoder (CEU).
5. Séquence finale normalisée à une longueur T fixée.

### 3. Encodage Audio

Audio réel (MFCC) :
- Extraction MFCC.
- Échantillonnage temporel sur T frames.
- Projection linéaire vers la dimension du modèle.

Audio textuel (AED) :
- Extraction pitch, loudness, jitter, shimmer.
- Catégorisation.
- Création d’une phrase descriptive.
- Encodage via BERT + CEU.

### 4. Vision Symbolique (VED)

- Mapping Emotion → Action Units.
- Création d’une phrase descriptive.
- Encodage via BERT.

### 5. Construction des représentations modales

- H_t0 = fc_text([X_t, D_a, D_v])
- H_a0 = fc_audio([X_a, D_a])
- H_v0 = fc_vision([X_v, D_v])
- H_t = CEU(H_t0)

### 6. Fusion Multimodale (MFU + CMA)

1. Initialisation de la fusion multimodale.
2. Application de plusieurs couches MFU :
   - Att_t2a : attention texte → audio
   - Att_t2v : attention texte → vision
   - Mise à jour H_m = H_m + α*att_a + β*att_v
3. Cross-modal attention finale H_fused = CMA(H_t, H_m)
4. Pooling temporel.

### 7. Tête de régression

- MLP à deux couches.
- Sortie : score réel du sentiment.
- MSELoss.

### 8. Métriques

- Acc-2, Acc-3, Acc-5, Acc-7.
- F1 score.
- MAE.
- Corrélation de Pearson.

---

## Correction du problème du tanh() et saturation du gradient

### 1. Problème initial

La tête du modèle utilisait :

y_pred = torch.tanh(self.regressor(pooled))

Conséquences :
- Sortie bornée dans [-1,1].
- Incapacité à atteindre les vraies valeurs des labels (ex : MOSI [-3,+3]).
- Saturation du gradient du tanh → apprentissage quasi nul.
- Plateau des métriques.

### 2. Correction apportée

Nous avons remplacé :
```
y_pred = torch.tanh(self.regressor(pooled))
```
par :
```
y_pred = self.regressor(pooled)
```


### 3. Effets observés

- Alignement de l’échelle des prédictions avec les labels.
- Disparition de la saturation des gradients.
- Baisse significative du MAE.
- Hausse importante des accuracies.
- Corrélation passant de ~0 à ~0.85.
- Courbes d'apprentissage redevenues normales.

### 4. Interprétation

La régression doit prédire une valeur réelle non bornée.  
Le tanh imposait une contrainte artificielle incompatible avec les labels et bloquait l’apprentissage.  
La suppression du tanh est la principale raison de l’amélioration drastique des performances du modèle.



## Résultats Résumés

### MELD – Modèle DEVA-like

| Metric | Score |
|--------|-------|
| ACC-2 | 0.94 |
| ACC-3 | 0.93 |
| ACC-5 | 0.86 |
| ACC-7 | 0.86 |
| F1 | 0.94 |
| MAE | 0.26 |
| Corr | 0.85 |
| Train Loss | 0.17 |

---

### MOSI – Baseline multimodale

| Metric | Score |
|--------|-------|
| ACC-2 | 0.86 |
| ACC-5 | 0.56 |
| ACC-7 | 0.76 |
| F1 | 0.85 |
| MAE | 0.47 |
| Corr | 0.95 |

## Exécution
lancer dans google collab :

MOSI_Deeplearning.ipynb 

MELD_projet.ipynb


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
