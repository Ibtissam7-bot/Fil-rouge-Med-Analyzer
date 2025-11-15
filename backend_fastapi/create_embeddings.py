import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import re
import time

print("Démarrage du script d'indexation...")

# --- 1. Chargement des Données (Base de connaissances) ---
try:
    df = pd.read_csv('C:/Users/Infinix/Desktop/Projet fil rouge/data/processed/medical_dataset_ml.csv')
except FileNotFoundError:
    print("Erreur: Fichier 'data/processed/medical_dataset_ml.csv' non trouvé.")
    exit()

def clean_symptoms_for_nlp(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[_,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['symptoms_processed'] = df['All_Symptoms'].apply(clean_symptoms_for_nlp)
print(f"Base de connaissances chargée : {df.shape[0]} cas.")

# --- 2. Chargement du "Cerveau" (Modèle Sémantique) ---
# (S'il est déjà en cache, c'est très rapide)
print("Chargement du modèle SentenceTransformer...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Modèle chargé.")

# --- 3. Vectorisation (L'étape LENTE) ---
corpus_symptoms = df['symptoms_processed'].tolist()

print("Début de la création des embeddings (cela peut prendre une minute)...")
start_time = time.time()
corpus_embeddings = model.encode(corpus_symptoms, convert_to_tensor=True, show_progress_bar=True)
end_time = time.time()

print(f"Embeddings créés en {end_time - start_time:.2f} secondes.")
print(f"Shape des embeddings : {corpus_embeddings.shape}")

# --- 4. SAUVEGARDE ! ---
# C'est ici la magie. On sauvegarde le résultat de l'étape lente.
# On sauvegarde aussi la dataframe nettoyée pour lier les index.
output_file = "corpus_embeddings.pt"
df.to_pickle("processed_dataframe.pkl")
torch.save(corpus_embeddings, output_file)

print(f"--- ✅ Terminé ! ---")
print(f"Embeddings sauvegardés dans : {output_file}")
print(f"DataFrame sauvegardée dans : processed_dataframe.pkl")