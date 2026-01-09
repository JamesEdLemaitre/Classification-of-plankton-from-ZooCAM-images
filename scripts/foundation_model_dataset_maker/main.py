import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION UTILISATEUR ---

DATASET_PATHS = {
    "challenge": "data/original_dataset",
    "pelgas": "data/pelgas_dataset/101141/individual_images",   
    "whoi":      "data/whoi_dataset",      
    "ndsb":     "data/ndsb_dataset",    
}

MIX_CONFIGS = {
    # Debug léger 
    "DEBUG": {
        "challenge": 1000, 
        "pelgas": 1000,   
        "whoi":      1000,
        "ndsb":     1000,
        "seed":      42
    },
    # Production : Volumes réels
    "PRODUCTION": {
        "challenge": "all",
        "pelgas": 1_000_000,        
        "whoi":      481_082,   
        "ndsb":     30_336,     
        "seed":      2024
    }
}

IMG_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

# -----------------------------------------------------------

def get_all_images(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ ERREUR: Le dossier '{folder_path}' n'existe pas.")
        return []
    files = []
    for ext in IMG_EXTENSIONS:
        found = glob.glob(os.path.join(folder_path, '**', ext), recursive=True)
        files.extend(found)
    return files

def index_dataset(name, root_path):
    print(f"   📂 Indexation source: {name.upper()} ({root_path})...")
    data_records = []

    if name == "challenge":
        # Train challenge -> Trainable
        train_dir = os.path.join(root_path, 'train')
        if os.path.exists(train_dir):
            imgs = get_all_images(train_dir)
            for p in imgs:
                data_records.append({'filepath': p, 'source': name, 'split_role': 'trainable'})

        # Test challenge -> Validation Only
        test_dir = os.path.join(root_path, 'test')
        if os.path.exists(test_dir):
            imgs = get_all_images(test_dir)
            for p in imgs:
                data_records.append({'filepath': p, 'source': name, 'split_role': 'validation_only'})
    else:
        # Autres -> Tout est Trainable
        imgs = get_all_images(root_path)
        for p in imgs:
            data_records.append({'filepath': p, 'source': name, 'split_role': 'trainable'})

    df = pd.DataFrame(data_records)
    print(f"      -> {len(df)} images trouvées.")
    return df

def sample_source(df, n_target, seed=42):
    if df.empty: return df
    count = len(df)
    
    if n_target == "all": return df.copy()
    if n_target >= count: return df.copy()

    return df.sample(n=int(n_target), random_state=seed)

def generate_mixed_dataset(full_indexes, config_name, config, output_dir):
    print(f"\n🏗️  Génération de la configuration : [ {config_name} ]")
    selected_dfs = []

    for source_name, count_target in config.items():
        if source_name not in full_indexes: continue

        df_source = full_indexes[source_name]
        
        # Séparation
        df_trainable = df_source[df_source['split_role'] == 'trainable']
        df_val_only = df_source[df_source['split_role'] == 'validation_only']

        # 1. Sampling de la partie Entraînable
        df_sampled_train = sample_source(df_trainable, count_target, seed=config.get('seed', 42))

        # 2. Sampling intelligent de la partie Validation Only (CORRECTION ICI)
        # Si on est en mode DEBUG (count_target != "all"), on réduit aussi le validation set
        if count_target != "all" and not df_val_only.empty:
            # Règle : on ne garde pour la validation externe que 20% de la quantité demandée en train
            # Ex: Si on demande 5000 images train, on garde max 1000 images validation externe
            val_limit = int(count_target * 0.2)
            val_limit = max(50, val_limit) # Sécurité min
            
            df_val_only = sample_source(df_val_only, val_limit, seed=config.get('seed', 42))
            print(f"      (Validation restreinte à {len(df_val_only)} images pour {source_name})")

        final_source_df = pd.concat([df_sampled_train, df_val_only])
        selected_dfs.append(final_source_df)
        
        print(f"   -> Ajout {source_name.upper()}: {len(df_sampled_train)} trainables + {len(df_val_only)} fixés")

    # Fusion globale
    master_df = pd.concat(selected_dfs, ignore_index=True)

    # Split Final
    df_trainable = master_df[master_df['split_role'] == 'trainable']
    df_forced_val = master_df[master_df['split_role'] == 'validation_only']

    # 10% de validation interne sur les données 'trainable'
    if len(df_trainable) > 0:
        actual_train, val_internal = train_test_split(
            df_trainable, 
            train_size=0.90, 
            random_state=config.get('seed', 42),
            shuffle=True
        )
    else:
        actual_train, val_internal = df_trainable, pd.DataFrame()

    # On combine les validations
    final_val = pd.concat([val_internal, df_forced_val])
    final_train = actual_train

    # Shuffle final
    final_train = final_train.sample(frac=1, random_state=config['seed']).reset_index(drop=True)
    final_val = final_val.sample(frac=1, random_state=config['seed']).reset_index(drop=True)

    # Sauvegarde
    train_path = os.path.join(output_dir, f"foundation_train_{config_name.lower()}.csv")
    val_path = os.path.join(output_dir, f"foundation_val_{config_name.lower()}.csv")

    final_train[['filepath']].to_csv(train_path, index=False)
    final_val[['filepath']].to_csv(val_path, index=False)

    print(f"\n   ✅ Fichiers générés dans {output_dir}")
    print(f"      Train : {len(final_train):9d} images")
    print(f"      Val   : {len(final_val):9d} images")
    print("-" * 50)

def main():
    OUTPUT_DIR = "foundation_data_config_splits"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_indexes = {}
    print("--- 1. Scan des disques ---")
    for name, path in DATASET_PATHS.items():
        full_indexes[name] = index_dataset(name, path)

    if sum([len(df) for df in full_indexes.values()]) == 0:
        print("❌ Aucune image trouvée.")
        return

    print("\n--- 2. Génération des Splits ---")
    for config_name, config in MIX_CONFIGS.items():
        generate_mixed_dataset(full_indexes, config_name, config, OUTPUT_DIR)

if __name__ == "__main__":
    main()
