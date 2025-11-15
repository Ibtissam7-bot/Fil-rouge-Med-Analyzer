import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware # IMPORT pour CORS
import time
import os

# --- Configuration de l'Application ---
app = FastAPI(
    title="MedAnalyzer AI API",
    description="API pour le diagnostic médical différentiel.",
    version="1.0.0"
)

# --- Configuration CORS (TRÈS IMPORTANT) ---
# Autorise ton application React (sur localhost:3000)
# à parler à ton API (sur localhost:8000)
origins = [
    "http://localhost:3000",
    "http://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Autorise POST, GET, etc.
    allow_headers=["*"],
)

# --- Chargement au Démarrage ---
# Cette fonction spéciale est appelée 1 SEULE FOIS, quand uvicorn démarre.
# C'est ici qu'on charge les modèles en mémoire (l'étape de 6.9s)
@app.on_event("startup")
async def load_models_on_startup():
    print("--- Démarrage du serveur : Chargement des modèles... ---")
    app.state.model_assets = {} # Dictionnaire pour stocker les modèles
    
    # 1. Charger le "Cerveau"
    app.state.model_assets["transformer"] = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 2. Charger la "Base de Connaissances" (les fichiers .pt et .pkl)
    try:
        app.state.model_assets["dataframe"] = pd.read_pickle("processed_dataframe.pkl")
        app.state.model_assets["embeddings"] = torch.load("corpus_embeddings.pt")
        print("--- Modèles et données chargés. API Prête. ---")
    except FileNotFoundError:
        print("ERREUR: Fichiers .pkl ou .pt non trouvés. Lancez create_embeddings.py d'abord.")
        app.state.model_assets = None # Marque comme échoué

# --- Modèles de Données (Le "Contrat" API) ---
# L'équivalent de 'interface' en TypeScript
# Il dit à FastAPI : "Le JSON entrant DOIT avoir un champ 'symptoms_query'"
class SymptomQuery(BaseModel):
    symptoms_query: str

# Il dit à FastAPI : "La réponse que je renvoie AURA cette structure"
class DiagnosisResult(BaseModel):
    rank: int
    disease: str
    confidence_score: float
    matched_symptoms_case: str

# --- Le "Endpoint" de Diagnostic (Le Cœur de l'API) ---
# C'est l'URL que React va appeler
@app.post("/diagnose", response_model=list[DiagnosisResult])
async def diagnose_patient(query: SymptomQuery):
    """
    Accepte les symptômes en texte libre et retourne 
    les 3 diagnostics différentiels les plus probables.
    """
    start_pred_time = time.time() # Timer pour vérifier les < 200ms

    # Vérifier si le démarrage a réussi
    if not app.state.model_assets:
        raise HTTPException(status_code=503, detail="Service non dispo (modèles non chargés)")

    # Récupérer les assets chargés au démarrage (rapide)
    model = app.state.model_assets["transformer"]
    df = app.state.model_assets["dataframe"]
    corpus_embeddings = app.state.model_assets["embeddings"]

    # --- L'inférence (LA PRÉDICTION RAPIDE) ---
    # 1. Encoder la requête (texte libre) de l'utilisateur
    query_embedding = model.encode(query.symptoms_query, convert_to_tensor=True)

    # 2. Chercher les 3 plus proches dans la base de connaissances
    search_results = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)
    # --- Fin de l'inférence ---

    # 3. Formater la réponse pour la rendre propre
    response_list = []
    for i, result in enumerate(search_results[0]):
        corpus_index = result['corpus_id']
        similarity_score = result['score']
        matched_case = df.iloc[corpus_index]
        
        response_list.append(DiagnosisResult(
            rank=i + 1,
            disease=matched_case['Disease'],
            confidence_score=round(similarity_score * 100, 2),
            matched_symptoms_case=matched_case['symptoms_processed']
        ))
    
    end_pred_time = time.time()
    print(f"Temps de réponse de la prédiction : {(end_pred_time - start_pred_time) * 1000:.2f} ms")
    
    return response_list