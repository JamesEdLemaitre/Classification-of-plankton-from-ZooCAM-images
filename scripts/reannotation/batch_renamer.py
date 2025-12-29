import os
from pathlib import Path
from tqdm import tqdm

# =========================================================================
# ⚙️ CONFIGURATION
# =========================================================================

# Dossier racine contenant tes sous-dossiers de classes
# Ex structure :
# data/train/
#    ├── 1_Copepoda/
#    ├── 2_Gelatinous/
#    └── 049_Complex_Name_Blabla/
ROOT_DIR = "/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi_cleaned/kaggle_challenge_sdi/data/reannotated_original_dataset/train"

# Liste des fichiers à traiter
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.bmp'}

# ⚡ EXCEPTIONS : Mappage manuel
# Si le dossier est la clé, on utilise la valeur comme préfixe.
# Sinon, on utilise le nom du dossier par défaut.
CUSTOM_RULES = {
    # "Nom_Du_Dossier_Reel": "Prefixe_Custom_Voulu"
    "0021_other_Siphonophorae_elongated": "002_other_Siphonophorae", 
    "0022_other_Siphonophorae_gelatinous": "002_other_Siphonophorae", 
    "0023_other_Siphonophorae_spherical": "002_other_Siphonophorae", 
    "0301_larvae_Annelida_Hard_Shell_Like": "030_larvae_Annelida",
    "0302_larvae_Annelida_Elongated_Like": "030_larvae_Annelida",
    "0491_Cnidaria_Hard_Shell_like": "049_Cnidaria",
    "0492_Cnidaria_Gelatinous": "049_Cnidaria",
}

# =========================================================================
# 🚀 MAIN
# =========================================================================

def list_files(root):
    # Récupère tous les fichiers dans les sous-dossiers (profondeur 1)
    p = Path(root)
    # Glob recursive pour attraper toutes les images
    files = [f for f in p.rglob('*') if f.suffix.lower() in EXTENSIONS and f.is_file()]
    return files

def main():
    print(f"📡 Scan du dossier : {ROOT_DIR}")
    
    files = list_files(ROOT_DIR)
    total_files = len(files)
    
    if total_files == 0:
        print("❌ Aucun fichier trouvé. Vérifie le chemin.")
        return

    print(f"📂 {total_files} fichiers trouvés. Démarrage du renommage...")

    renamed_count = 0
    skipped_count = 0

    # Barre de progression
    for file_path in tqdm(files, desc="Renommage", unit="img"):
        
        # 1. Identifier le dossier parent (la classe)
        folder_name = file_path.parent.name
        
        # 2. Déterminer le préfixe
        if folder_name in CUSTOM_RULES:
            prefix = CUSTOM_RULES[folder_name]
        else:
            prefix = folder_name
            
        # 3. Construire le nouveau nom
        # Format : Prefix_NomOriginal
        original_name = file_path.name
        
        # SÉCURITÉ : Vérifier si le fichier commence déjà par le préfixe (pour ne pas le refaire 2 fois)
        expected_start = f"{prefix}_"
        if original_name.startswith(expected_start):
            skipped_count += 1
            continue

        new_name = f"{prefix}_{original_name}"
        new_path = file_path.parent / new_name

        # 4. Renommage effectif
        try:
            # .rename() est atomique sur la plupart des OS
            file_path.rename(new_path)
            renamed_count += 1
        except OSError as e:
            print(f"\n❌ Erreur sur {original_name} : {e}")

    print("\n--- TERMINÉ ---")
    print(f"✅ Renommés : {renamed_count}")
    print(f"⏩ Ignorés (déjà bien nommés) : {skipped_count}")

if __name__ == "__main__":
    main()
