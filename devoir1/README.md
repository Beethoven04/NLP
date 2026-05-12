# Devoir 1 — Code de la route NLP pipeline

Extraction + structuration des 126 pages du `code de la route MA52_05.pdf` marocain. Produit `export_final.csv` (126 articles) qui sert d'entrée au devoir 2.

## What it does

- Lit toutes les pages du PDF (`PyPDF2`).
- Normalise le texte arabe (tashkeel, alef variants, espaces, marqueurs directionnels).
- Détecte les frontières d'articles (`المادة N`, `الماده N`, variantes OCR) et garde la page d'origine.
- Extrait : `infraction_desc`, `sanction_desc`, `amende_fixe`, `points_retrait`, `mots_cles`, `role`, `type_infraction`, `niveau_gravite`, `localisation`.
- Clustering TF-IDF (KMeans) + entités spaCy si le modèle `xx_ent_wiki_sm` est installé (silencieusement ignoré sinon).
- Export CSV final UTF-8 BOM, lisible dans Excel.

## Files

| File | Purpose |
|---|---|
| `devoir1_pipeline_kaggle.ipynb` | Notebook principal (à rendre). |
| `code de la route MA52_05.pdf` | PDF source (4.4 MB). |
| `export_final.csv` | Sortie : 126 articles structurés. |
| `devoir1_extracted.txt` | Extrait texte brut (intermédiaire). |
| `requirements.txt` | Pointe vers `../requirements.txt`. |

## Run

```bash
cd devoir1
jupyter notebook devoir1_pipeline_kaggle.ipynb
```

Le notebook tourne complètement sans clé API. spaCy est optionnel — si `xx_ent_wiki_sm` n'est pas installé, la colonne `spacy_entities` reste vide et le reste fonctionne normalement.
