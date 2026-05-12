# Devoir 0 — Arabic NLP basics

Pipeline NLP de base sur un texte arabe : chargement, normalisation, tokenisation, stopwords, segmentation en phrases, TF/IDF/TF-IDF, similarité cosinus.

## Files

| File | Purpose |
|---|---|
| `devoir0.ipynb` | Notebook principal (à rendre). |
| `devoir1.ipynb` | Copie de l'ancien nom — à supprimer avant push. |
| `arabic_nlp_text.txt` | Corpus source. |
| `requirements.txt` | Pointe vers `../requirements.txt`. |

## Run

```bash
cd devoir0
jupyter notebook devoir0.ipynb
```

Le notebook télécharge la liste NLTK des stopwords arabes au premier lancement. S'il n'a pas accès au réseau, une liste embarquée prend le relais (cellule 7).
