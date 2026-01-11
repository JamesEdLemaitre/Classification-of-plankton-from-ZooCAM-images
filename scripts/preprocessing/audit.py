import os
import cv2
import random
from pathlib import Path
import pandas as pd

# === CONFIGURATION ===
SOURCE_DIR = "/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi_cleaned/kaggle_challenge_sdi/data/whoi_dataset"

# Dossier où sera créé le dataset optimisé
DEST_DIR = "/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi_cleaned/kaggle_challenge_sdi/data/whoi_dataset_optimized_512px"

# =====================

def get_file_pair(src_root, dest_root):
    src_path = Path(src_root)
    dest_path = Path(dest_root)
    
    pairs = []
    
    # On regarde ce qu'il y a dans le dossier de destination
    dest_files = list(dest_path.rglob("*"))
    dest_files = [f for f in dest_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif']]
    
    print(f"Analyse d'un échantillon parmi {len(dest_files)} fichiers...")
    random.shuffle(dest_files)
    
    # On prend 10 images au hasard pour l'audit
    analyzed = 0
    data = []
    
    for d_file in dest_files:
        if analyzed >= 15: break
        
        # Trouver la source correspondante
        rel_path = d_file.relative_to(dest_path)
        
        # Astuce : La source peut avoir une extension différente (.png vs .jpg)
        # On cherche le fichier source avec le même nom de base
        s_candidate = src_path / rel_path
        
        found_src = None
        if s_candidate.exists():
            found_src = s_candidate
        else:
            # Si l'extension a changé (ex: png -> jpg), on cherche l'original
            stem = rel_path.stem
            parent = rel_path.parent
            possible_exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
            for ext in possible_exts:
                potential = src_path / parent / (stem + ext)
                if potential.exists():
                    found_src = potential
                    break
        
        if found_src:
            # Analyse
            img_s = cv2.imread(str(found_src))
            img_d = cv2.imread(str(d_file))
            
            if img_s is None or img_d is None: continue

            h_s, w_s = img_s.shape[:2]
            h_d, w_d = img_d.shape[:2]
            
            size_s = os.path.getsize(found_src)
            size_d = os.path.getsize(d_file)
            
            # Calculs
            pixels_s = h_s * w_s
            pixels_d = h_d * w_d
            
            pixel_reduction = pixels_s / pixels_d if pixels_d > 0 else 1
            size_reduction = size_s / size_d if size_d > 0 else 1
            
            action = "COPIE (Identique)"
            if abs(pixels_s - pixels_d) > 100: # Marge d'erreur
                action = "RESIZE"
            
            data.append({
                "Fichier": d_file.name,
                "Action": action,
                "Orig Dim": f"{w_s}x{h_s}",
                "New Dim": f"{w_d}x{h_d}",
                "Pixel Factor": f"x{pixel_reduction:.1f}",
                "Size Factor": f"x{size_reduction:.1f}",
                "Orig Size (Ko)": int(size_s/1024),
                "New Size (Ko)": int(size_d/1024)
            })
            analyzed += 1

    return data

if __name__ == "__main__":
    results = get_file_pair(SOURCE_DIR, DEST_DIR)
    
    print("\n--- AUDIT COMPARATIF (Echantillon aléatoire) ---")
    # Affichage tabulaire propre
    df = pd.DataFrame(results)
    if not df.empty:
        # On réorganise les colonnes pour la lisibilité
        cols = ["Action", "Orig Dim", "New Dim", "Pixel Factor", "Size Factor", "Orig Size (Ko)", "New Size (Ko)"]
        print(df[cols].to_string(index=False))
    else:
        print("Aucune paire trouvée pour comparaison.")

    print("\n\nINTERPRETATION :")
    print("1. Si 'Pixel Factor' est proche de 1.0 : L'image n'a pas été redimensionnée (Copie).")
    print("2. Si 'Size Factor' >> 'Pixel Factor' : La compression JPEG aide beaucoup.")
    print("3. Si 'Size Factor' ≈ 'Pixel Factor' : Le gain vient purement de la réduction de taille.")
