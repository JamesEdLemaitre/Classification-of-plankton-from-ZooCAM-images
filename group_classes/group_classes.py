import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

# Chemin vers tes données (doit contenir les sous-dossiers 000_..., 001_...)
DATA_DIR = './data/train' 

# Dossier où seront enregistrées les images de validation
OUTPUT_DIR = './group_classes/validation_groupes_hybrides'

# ==========================================
# DÉFINITION DU DÉCOUPAGE HYBRIDE (CORRIGÉ V2)
# ==========================================

GROUP_MAPPING = {
    "1_Copepoda_Torpille": [
        "003_Harpacticoida", "010_Oncaea", "011_Euchaeta", "013_Acartiidae",
        "015_Metrinidae", "017_Poecilostomatoida", "018_Calocalanus", "021_Corycaeidae",
        "029_Temoridae", "033_Calanoida", "035_Calanidae", "041_Euterpina",
        "044_Cyclopoida", "050_Pontellidae", "057_Candaciidae", "059_Centropagidae",
        "066_multiple_Copepoda", "068_Copepoda", "078_Microstella", "081_Isias",
        # Ajout suite modif : Neoceratium a des "cornes" rappelant les antennes copepodes
        "031_Neoceratium" 
    ],
    
    "2_Gelatinous_Transparent": [
        "000_Physonectae", "002_other_Siphonophorae",
        "005_comb_Ctenophora", "014_bract_Diphyidae", 
        "032_nectophore_Diphyidae", "034_Doliolida", "042_gelatinous",
        "043_Thaliacea", "049_Cnidaria", "054_gonophore_Diphyidae",
        "064_Hydrozoa", "071_Obelia", "075_eudoxie_Diphyidae", "076_Ctenophora",
        # Ajout suite modif : Les larves d'echinodermes sont souvent vitreuses/transparentes
        "006_larvae_Echinodermata",
        # Pas parfait, mais globalement translucide:
        "061_Luidiidae",
    ],
    # 001 et 019 déplacés vers Allongés
    
    "3_Macro_Crustacea_Larvae": [
        "016_Ciripedia", "022_Decapoda", "036_shrimp_like", "039_other_Crustacea",
        "040_megalopa", "045_Amphipoda", "053_nauplii_Crustacea", "058_Nannosquillidae",
        "065_larvae_Crustacea", "070_zoea_brachyura", "085_Porcellanidae",
        # Ajout suite modif
        "083_part_Crustacea"
    ],
    
    "4_Spherical_Compact_Eggs": [
        "007_Noctiluca", "026_egg_Mollusca", "038_egg_Engraudilae",
        "048_egg_other", "051_Halosphaera", "055_egg_Sardina", 
        "084_egg_Actinopterygii",
        # Cladocères (compacts/ronds)
        "027_Evadne", "052_Cladocera", "067_Penilia", "077_Podon",
        # Ajouts suite modif (Mollusques & formes rondes)
        "012_Diatoma_tenuis", "025_Bivalvia", "056_Limacinidae", "060_Gastropoda"
        # Ajout perso depuis elongated
        "072_Diatoma",
    ],
    
    "5_Elongated_and_Chains": [
        "020_chainlarge", "024_Chaetognatha", "028_Actinopterygii", 
        "030_larvae_Annelida", "062_Rhizosolenids",
         "074_Thalassionema", "080_Trichodesmium", "082_Annelida",
        # Ajout suite modif (Appendiculaires = queue allongée)
        "001_tail_Appendicularia", "019_Appendicularia"
    ],
    # 031 déplacé vers Copepoda, 012 vers Sphérique
    
    "6_Hard_Shells_Misc": [
        # Groupe réduit, contient le reste des formes dures/géométriques non rondes
        "023_Rhizaria", "047_pluteus_Echinodermata",
         "073_Phoronida", "079_veliger"
    ],
    
    "7_Detritus_Noise_Empty": [
        "004_empty_Harpacticoida", "008_detritus", "009_empty_Copepoda",
        "037_feces", "046_other_living"
    ],
    # 083 déplacé vers Crustacea
    
    "8_Triangulaire_Allonge": [
        # Nouvelle classe créée pour ces formes spécifiques (triangle isocèle)
        "063_Cavoliniidae", "069_cyphonaute"
    ]
}


def visualize_groupings(data_path, output_path, mapping):
    """
    Génère une image composite pour chaque groupe du mapping.
    """
    base_path = Path(data_path)
    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"--- Début de la génération ---")
    print(f"Lecture depuis : {base_path}")
    print(f"Sauvegarde dans : {out_path}\n")

    for group_name, subclasses in mapping.items():
        print(f"Traitement du groupe : {group_name} ({len(subclasses)} classes)")
        
        # Collecter une image par sous-classe
        images_found = []
        labels_found = []

        for sub_cls in sorted(subclasses):
            folder = base_path / sub_cls
            
            # Vérifier si le dossier existe
            if not folder.exists():
                print(f"  [AVERTISSEMENT] Dossier introuvable : {sub_cls}")
                continue
            
            # Prendre la première image (jpg, png, tif, etc.)
            files = list(folder.glob('*.*'))
            # Filtrer pour garder les images communes
            valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
            img_file = next((f for f in files if f.suffix.lower() in valid_exts), None)
            
            if img_file:
                images_found.append(img_file)
                labels_found.append(sub_cls)
            else:
                print(f"  [INFO] Aucune image dans : {sub_cls}")

        if not images_found:
            print(f"  -> Groupe vide, image ignorée.")
            continue

        # --- Création de la grille Matplotlib ---
        num_imgs = len(images_found)
        cols = 5  # Nombre de colonnes arbitraire pour la lisibilité
        rows = math.ceil(num_imgs / cols)

        fig_width = 4 * cols
        fig_height = 4 * rows + 2 # +2 pour le titre
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        fig.suptitle(f"Groupe : {group_name}", fontsize=20, weight='bold')
        
        # Aplatir les axes pour itérer facilement
        axes_flat = axes.flatten() if num_imgs > 1 else [axes]

        for i, (img_path, label) in enumerate(zip(images_found, labels_found)):
            try:
                img = mpimg.imread(str(img_path))
                ax = axes_flat[i]
                ax.imshow(img, cmap='gray') # Souvent plancton en niveau de gris
                ax.set_title(label[:20], fontsize=9) # Tronquer si trop long
                ax.axis('off')
            except Exception as e:
                print(f"  Erreur lecture {img_path}: {e}")

        # Cacher les axes vides restants
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuster pour le titre
        save_filename = out_path / f"Visualisation_{group_name}.png"
        plt.savefig(save_filename)
        plt.close(fig) # Libérer la mémoire
        print(f"  -> Image générée : {save_filename}")

    print("\n--- Terminé ! ---")
    print(f"Vérifie les résultats dans le dossier : {output_path}")

if __name__ == "__main__":
    visualize_groupings(DATA_DIR, OUTPUT_DIR, GROUP_MAPPING)

