import os
import cv2
import shutil
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import time

# ================= CONFIGURATION =================
# Dossier contenant tes images (ex: tes crops)
SOURCE_DIR = "/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi_cleaned/kaggle_challenge_sdi/data/original_dataset/test"

# Dossier où sera créé le dataset optimisé
DEST_DIR = "/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi_cleaned/kaggle_challenge_sdi/data/original_dataset_optimized_512px/test"



TARGET_SIZE = 512 
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
NUM_WORKERS = None 

# Qualité de sortie pour les images QUI SONT REDIMENSIONNÉES uniquement
# 85 est le standard industriel pour le ML (bon ratio poids/qualité)
JPEG_QUALITY = 85 
# =================================================

def format_bytes(size):
    power = 2**10
    n = size
    power_labels = {0 : '', 1: 'Ko', 2: 'Mo', 3: 'Go', 4: 'To'}
    n_iter = 0
    while n > power and n_iter < 4: # Safety limit
        n /= power
        n_iter += 1
    return f"{n:.2f} {power_labels[n_iter]}"

def process_image(args):
    src_path, dst_path, target_max = args
    
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        orig_size = os.path.getsize(src_path)
        
        # Lecture juste pour vérifier les dimensions (ou header)
        img = cv2.imread(src_path)
        if img is None:
            return (0, 0, 0)

        h, w = img.shape[:2]
        max_dim = max(h, w)
        
        # --- CAS 1 : IMAGE TROP GRANDE -> RESIZE & COMPRESSION ---
        if max_dim > target_max:
            scale = target_max / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # On force l'extension .jpg pour uniformiser la sortie des images traitées
            base, _ = os.path.splitext(dst_path)
            dst_path_jpg = base + ".jpg"
            
            cv2.imwrite(dst_path_jpg, resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            
            final_size = os.path.getsize(dst_path_jpg)
            return (2, orig_size, final_size)

        # --- CAS 2 : IMAGE PETITE/OK -> COPIE BRUTE (FILE SYSTEM COPY) ---
        else:
            # ICI LE CHANGEMENT : On ne touche pas aux pixels, on copie le fichier binaire.
            # On garde le poids original exact.
            shutil.copy2(src_path, dst_path)
            
            final_size = os.path.getsize(dst_path)
            return (1, orig_size, final_size)

    except Exception as e:
        return (0, 0, 0)

def main():
    start_time = time.time()
    source_path = Path(SOURCE_DIR)
    dest_path = Path(DEST_DIR)
    
    print(f"--- OPTIMISATION DATASET (Smart Copy) ---")
    print(f"Source : {source_path}")
    print(f"Cible  : {dest_path}")
    print(f"Stratégie : Resize si > {TARGET_SIZE}px, sinon Copie brute")
    print("-" * 30)
    
    # 1. Lister
    print("Inventaire...")
    all_files = []
    for ext in EXTENSIONS:
        all_files.extend(list(source_path.rglob(f"*{ext}")))
        all_files.extend(list(source_path.rglob(f"*{ext.upper()}")))
    all_files = list(set(all_files))
    print(f"{len(all_files)} images trouvées.")
    
    tasks = []
    for f in all_files:
        rel_path = f.relative_to(source_path)
        target_f = dest_path / rel_path
        tasks.append((str(f), str(target_f), TARGET_SIZE))
        
    # 2. Exécution
    stats = {'processed': 0, 'resized': 0, 'untouched': 0, 'errors': 0, 
             'total_orig_size': 0, 'total_new_size': 0}
    
    # ProcessPoolExecutor est nécessaire car cv2.imread libère le GIL
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_image, tasks, chunksize=50), total=len(tasks), unit="img"))
        
    for status, orig_s, new_s in results:
        if status == 0:
            stats['errors'] += 1
        else:
            stats['processed'] += 1
            stats['total_orig_size'] += orig_s
            stats['total_new_size'] += new_s
            if status == 2: stats['resized'] += 1
            elif status == 1: stats['untouched'] += 1

    duration = time.time() - start_time
    saved_bytes = stats['total_orig_size'] - stats['total_new_size']
    
    summary_text = (
        f"--- RAPPORT FINAL ---\n"
        f"Durée : {duration:.1f} s\n\n"
        f"STATISTIQUES :\n"
        f"Images Totales      : {stats['processed']}\n"
        f"Images Résimées     : {stats['resized']} (JPG Q={JPEG_QUALITY})\n"
        f"Images Copiées      : {stats['untouched']} (taille inchangée)\n"
        f"Erreurs             : {stats['errors']}\n\n"
        f"ESPACE DISQUE :\n"
        f"Avant : {format_bytes(stats['total_orig_size'])}\n"
        f"Après : {format_bytes(stats['total_new_size'])}\n"
        f"GAIN  : {format_bytes(saved_bytes)}\n"
    )
    
    print("\n" + summary_text)
    
    with open(dest_path / "summary_v2.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

if __name__ == "__main__":
    main()
