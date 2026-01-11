import os
import numpy as np
from pathlib import Path
from skimage import io, color, filters, measure, morphology
from tqdm import tqdm # Pour la barre de progression
import warnings

# On ignore les warnings liés aux images basse  qualité/contrast
warnings.filterwarnings("ignore")

# === CONFIGURATION ===
SOURCE_DIR = "/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi_cleaned/kaggle_challenge_sdi/data/pelgas_dataset/101141/individual_images"  # <--- METS TON CHEMIN ICI
DEST_DIR = "/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi_cleaned/kaggle_challenge_sdi/data/pelgas_dataset_cropped/101141/individual_images"


# === RÉGLAGES OPTIMISÉS (V4) ===
MARGIN_RATIO = 0.10      # Marge réduite à 10%
BOTTOM_TOLERANCE = 20    # Exclusion bas (échelle)
NOISE_TOLERANCE = 5      # Sensibilité écart au fond
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

def smart_crop_background_deviation(img):
    """
    Détecte le plancton par écart de luminosité par rapport au fond (Deviation Method).
    Retourne: l'image croppée (ou None si vide).
    """
    # 1. Conversion et Normalisation
    if img.ndim == 3:
        gray = color.rgb2gray(img)
        # Force conversion 0-255 uint8
        if gray.max() <= 1.0: gray = (gray * 255).astype(np.uint8)
    else:
        gray = img
        if gray.dtype != np.uint8 and gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
    
    h, w = gray.shape

    # 2. Détection Echelle (Zone interdite)
    scale_top_limit = h
    try:
        thresh_rough = filters.threshold_otsu(gray)
        mask_rough = gray < thresh_rough
        mask_rough = morphology.binary_dilation(mask_rough, morphology.disk(6))
        lbl_rough = measure.label(mask_rough)
        
        for r in measure.regionprops(lbl_rough):
            minr, minc, maxr, maxc = r.bbox
            # Si un objet touche le bas de l'image (zone tolerance)
            if maxr >= h - BOTTOM_TOLERANCE:
                if minr < scale_top_limit: scale_top_limit = minr
        
        # Petite marge de sécurité au dessus de l'échelle
        if scale_top_limit < h: 
            scale_top_limit = max(0, scale_top_limit - 5)
    except:
        pass # Pas d'échelle détectée ou erreur, on garde toute la hauteur

    # 3. Détection Plancton (Background Deviation)
    # Estimation du fond sur la moitié haute (pour éviter l'échelle)
    bg_sample = gray[0:h//2, :] if h > 20 else gray
    if bg_sample.size == 0: return None # Image corrompue ou trop petite
    
    bg_value = np.median(bg_sample)
    threshold_value = bg_value - NOISE_TOLERANCE
    
    mask_fine = (gray < threshold_value)
    # On coupe tout ce qui est en dessous de la limite de l'échelle
    mask_fine[scale_top_limit:h, :] = False 
    
    # Dilatation pour unir les morceaux disjoints
    mask_plankton = morphology.binary_dilation(mask_fine, morphology.disk(2))

    # 4. Calcul du Crop
    coords = np.argwhere(mask_plankton)
    
    if coords.size == 0:
        return None # Rien détecté (image vide)

    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0)
    
    h_box, w_box = maxr - minr, maxc - minc
    
    # Application de la marge
    pad_h = int(h_box * MARGIN_RATIO)
    pad_w = int(w_box * MARGIN_RATIO)
    
    # Sécurisation des coordonnées
    safe_minr = max(0, minr - pad_h)
    safe_minc = max(0, minc - pad_w)
    
    # Le bas du crop ne doit JAMAIS dépasser l'échelle détectée
    safe_maxr = min(scale_top_limit, maxr + pad_h)
    safe_maxc = min(w, maxc + pad_w)
    
    # Vérification cohérence
    if safe_maxr <= safe_minr or safe_maxc <= safe_minc:
        return None

    # Extraction
    cropped_img = img[safe_minr:safe_maxr, safe_minc:safe_maxc]
    return cropped_img

def process_dataset():
    source_path = Path(SOURCE_DIR)
    dest_path = Path(DEST_DIR)

    # Récupérer toutes les images
    all_files = []
    print("Exploration des dossiers en cours...")
    for ext in EXTENSIONS:
        all_files.extend(list(source_path.rglob(f"*{ext}")))
        all_files.extend(list(source_path.rglob(f"*{ext.upper()}")))
    
    print(f"Extraction prévues : {len(all_files)} images.")
    print(f"Destination : {dest_path}")
    
    count_ok = 0
    count_skipped = 0
    
    # Barre de progression
    pbar = tqdm(all_files, unit="img")
    
    for file_path in pbar:
        try:
            # 1. Créer le dossier de destination correspondant
            relative_path = file_path.relative_to(source_path)
            target_file_path = dest_path / relative_path
            target_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 2. Charger l'image
            img = io.imread(str(file_path))
            
            # 3. Smart Crop
            cropped = smart_crop_background_deviation(img)
            
            # 4. Sauvegarder
            if cropped is not None and cropped.size > 0:
                # On évite de sauvegarder des crops minuscules (bruit)
                if cropped.shape[0] > 10 and cropped.shape[1] > 10:
                    io.imsave(str(target_file_path), cropped, check_contrast=False)
                    count_ok += 1
                else:
                    count_skipped += 1
            else:
                count_skipped += 1
                
            pbar.set_description(f"OK: {count_ok} | Vide: {count_skipped}")
            
        except Exception as e:
            # En cas de fichier corrompu, on continue
            # print(f"Erreur sur {file_path}: {e}")
            count_skipped += 1

    print("\n" + "="*40)
    print("TRAITEMENT TERMINÉ")
    print(f"Images conservées (Cropped) : {count_ok}")
    print(f"Images ignorées (Vides/Bruit) : {count_skipped}")
    print(f"Ratio de conservation : {count_ok / len(all_files):.2%}")
    print("="*40)

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"Erreur: Le dossier source n'existe pas: {SOURCE_DIR}")
    else:
        process_dataset()
