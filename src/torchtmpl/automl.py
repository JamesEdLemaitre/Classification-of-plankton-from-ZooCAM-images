import os
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.multimodal import MultiModalPredictor
import logging

# --- CONFIGURATION ---
DATA_DIR = "./data/data_reannotated_v2/train"
DATA_DIR = "./data/original_dataset/train"
OUTPUT_DIR = "autogluon_results"
METRIC = "f1_macro"
SEED = 42

# --- MODE TEST VS PRODUCTION ---
# Mets à True pour tester ton code en 5 min sur un petit bout du dataset
# Mets à False pour lancer le vrai entraînement de 10h
TEST_MODE = True 

if TEST_MODE:
    print("⚠️ MODE TEST ACTIVÉ : Utilisation d'un subset intelligent (5%) et temps limité.")
    TIME_LIMIT = 3600 * 8   
    SUBSET_RATIO = 0.30     # On garde seulement x% des données
else:
    print("🚀 MODE PRODUCTION : Utilisation du dataset complet.")
    TIME_LIMIT = 3600 * 10  # 10 heures
    SUBSET_RATIO = 1.0      # 100% des données

def prepare_dataframe(root_dir):
    """
    Parcourt les dossiers récursivement pour créer un DataFrame [image, label]
    """
    data = []
    print(f"🔍 Scan des images dans {root_dir}...")

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dossier introuvable : {root_dir}")

    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    print(f"   -> {len(classes)} classes trouvées.")

    for cls_name in classes:
        cls_folder = os.path.join(root_dir, cls_name)
        # On prend toutes les images courantes
        files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

        for fname in files:
            full_path = os.path.join(cls_folder, fname)
            data.append({
                'image': full_path,
                'label': cls_name
            })

    df = pd.DataFrame(data)
    print(f"✅ Total : {len(df)} images indexées.")
    return df

def main():
    # 1. Préparer les données
    df = prepare_dataframe(DATA_DIR)

    if df.empty:
        print("❌ Aucune image trouvée. Vérifie le chemin DATA_DIR.")
        return

    # 2. Split Train / Tuning (Validation)
    # D'abord on fait le split classique 85% Train / 15% Val
    train_data, tune_data = train_test_split(
        df, test_size=0.15, random_state=SEED, stratify=df['label']
    )

    # --- LOGIQUE DU SUBSET INTELLIGENT ---
    if TEST_MODE:
        print(f"\n✂️ CRÉATION DU SUBSET ({int(SUBSET_RATIO*100)}%)...")
        
        # On réduit le train_data en gardant la stratification
        # On essaie, mais si une classe a trop peu d'échantillons, sklearn peut râler.
        try:
            train_data, _ = train_test_split(
                train_data, 
                train_size=SUBSET_RATIO, 
                stratify=train_data['label'], # Garde les proportions exactes
                random_state=SEED
            )
            # On réduit aussi la validation pour que l'évaluation soit rapide
            tune_data, _ = train_test_split(
                tune_data, 
                train_size=SUBSET_RATIO, 
                stratify=tune_data['label'],
                random_state=SEED
            )
        except ValueError as e:
            print("⚠️ Attention : Impossible de stratifier parfaitement sur un si petit subset (certaines classes sont trop rares).")
            print("   -> Passage en mode aléatoire simple pour le test.")
            train_data = train_data.sample(frac=SUBSET_RATIO, random_state=SEED)
            tune_data = tune_data.sample(frac=SUBSET_RATIO, random_state=SEED)

    print(f"📊 Données utilisées pour ce run : {len(train_data)} train | {len(tune_data)} validation")

    # 3. Initialiser le Predictor
    predictor = MultiModalPredictor(
        label="label",
        problem_type="classification",
        path=OUTPUT_DIR,
        eval_metric=METRIC, 
    )

    # 4. Lancer l'entraînement (FIT)
    print(f"\n🚀 Lancement de l'AutoML pour {TIME_LIMIT} secondes...")
    
    predictor.fit(
        train_data=train_data,
        tuning_data=tune_data,
        time_limit=TIME_LIMIT,
        presets="medium_quality", # Compromis idéal Vitesse/Perf (ResNet/Swin Tiny)
        hyperparameters={
            "env.per_gpu_batch_size": 128, # Force un gros batch size pour profiter du GPU
        }
    )

    # 5. Évaluation finale
    print("\n🏁 Entraînement terminé.")
    scores = predictor.evaluate(tune_data, metrics=[METRIC, "accuracy"])
    print(f"📈 Résultats ({METRIC}) :")
    print(scores)

    print(f"\n💾 Modèle sauvegardé dans : {OUTPUT_DIR}")
    
    if TEST_MODE:
        print("\n✅ TEST RÉUSSI. Passe TEST_MODE = False pour lancer le gros entraînement.")

if __name__ == "__main__":
    main()
