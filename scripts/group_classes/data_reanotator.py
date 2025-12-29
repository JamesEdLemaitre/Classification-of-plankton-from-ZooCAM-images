import os
import shutil
from pathlib import Path
from tqdm import tqdm # Optionnel : pour la barre de chargement, sinon on fera des prints

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = Path('/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi_cleaned/kaggle_challenge_sdi/data/reannotated_original_dataset/train')
OUTPUT_DIR = Path('/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi_cleaned/kaggle_challenge_sdi//data/data_reannotated_v2/train')

# Liste des classes à ignorer (traitement manuel plus tard) ### NE PAS OUBLIER D'AJOUTER CES IMAGES MANUELLEMENT !
# IGNORE_PREFIXES = ["049"]
IGNORE_PREFIXES = []

# ==========================================
# LE DÉCOUPAGE VALIDÉ (V2)
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
        # modif du 28/12/2025:
        # Cladocères (compacts/ronds)
        "027_Evadne", "052_Cladocera", "067_Penilia", "077_Podon",
        # Anciennement détritus:
        "009_empty_Copepoda"
    ],
    
    # "2_Gelatinous_Transparent": [
    #     "000_Physonectae", "002_other_Siphonophorae",
    #     "005_comb_Ctenophora", "014_bract_Diphyidae", 
    #     "032_nectophore_Diphyidae", "034_Doliolida", "042_gelatinous",
    #     "043_Thaliacea", "049_Cnidaria", "054_gonophore_Diphyidae",
    #     "064_Hydrozoa", "071_Obelia", "075_eudoxie_Diphyidae", "076_Ctenophora",
    #     # Ajout suite modif : Les larves d'echinodermes sont souvent vitreuses/transparentes
    #     "006_larvae_Echinodermata",
    #     # Pas parfait, mais globalement translucide:
    #     "061_Luidiidae",
    # ],
    # 001 et 019 déplacés vers Allongés

    # Nouvelle version du 26/12/2025: 005 mis en allongé
     "2_Gelatinous_Transparent": [
        "000_Physonectae", "002_other_Siphonophorae",
        "014_bract_Diphyidae", 
        "032_nectophore_Diphyidae", "034_Doliolida", "042_gelatinous",
        "043_Thaliacea", "049_Cnidaria", "054_gonophore_Diphyidae",
        "064_Hydrozoa", "071_Obelia", "075_eudoxie_Diphyidae", "076_Ctenophora",
        # Ajout suite modif : Les larves d'echinodermes sont souvent vitreuses/transparentes
        "006_larvae_Echinodermata",
        # Pas parfait, mais globalement translucide:
        "061_Luidiidae",
        # Réannoté:
        "0492_Cnidaria_Gelatinous",
        "0021_other_Siphonophorae_gelatinous"
    ],
    
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
        # # Cladocères (compacts/ronds) => maintenant mis en 1 duis le 28/12/2025
        # "027_Evadne", "052_Cladocera", "067_Penilia", "077_Podon",
        # Ajouts suite modif (Mollusques & formes rondes)
        "012_Diatoma_tenuis", "025_Bivalvia", "056_Limacinidae", "060_Gastropoda"
        # Ajout perso depuis elongated
        "072_Diatoma",
    ],
    
    "5_Elongated_and_Chains": [
        "020_chainlarge", "024_Chaetognatha", "028_Actinopterygii", 
        # 030_larvae_Annelida", réannoté:
        "0302_larvae_Annelida_Elongated_Like", "062_Rhizosolenids",
        "074_Thalassionema", "080_Trichodesmium", "082_Annelida",
        # Ajout suite modif (Appendiculaires = queue allongée)
        "001_tail_Appendicularia", "019_Appendicularia",
        # modif du 26/12/2025: 
        "005_comb_Ctenophora",
        # réannoté:
        "0021_other_Siphonophorae_elongated",
        # modil du 28/12/2025, ajout depuis détritus:
        "037_feces"
    ],
    # 031 déplacé vers Copepoda, 012 vers Sphérique
    
    "6_Hard_Shells_Misc": [
        # Groupe réduit, contient le reste des formes dures/géométriques non rondes
        "023_Rhizaria", "047_pluteus_Echinodermata",
         "073_Phoronida", "079_veliger", # 79 je sais pas trop où le mettre
        # Réannoté:
        "0491_Cnidaria_Hard_Shell_Like",
        "0301_larvae_Annelida_Hard_Shell_Like"
    ],
    
    "7_Detritus_Noise_Empty": [
        "004_empty_Harpacticoida", "008_detritus", # "009_empty_Copepoda", supprimé, ressemble à Copepoda
        # "037_feces", supprimé et mis en allongé, les cacas sont souvent longs
        "046_other_living", # Gardé pour le moment mais preobablement qu'une réannotation en classes distinctes ne ferait pas de mal. 
        
    ],
    # 083 déplacé vers Crustacea
    
    "8_Triangulaire_Allonge": [
        # Nouvelle classe créée pour ces formes spécifiques (triangle isocèle)
        "063_Cavoliniidae", "069_cyphonaute"
    ]
}

def reorganize_data():
    # Création du dossier racine de sortie
    if not OUTPUT_DIR.exists():
        os.makedirs(OUTPUT_DIR)
        print(f"Dossier créé : {OUTPUT_DIR}")
    else:
        print(f"Dossier de sortie : {OUTPUT_DIR}")

    total_moved = 0
    
    # Itération sur les groupes définis (Classes Mères)
    for mother_class, subclasses in GROUP_MAPPING.items():
        print(f"\nTraitement du groupe : {mother_class.upper()}...")
        
        # Créer le dossier de la classe mère
        dest_mother_path = OUTPUT_DIR / mother_class
        os.makedirs(dest_mother_path, exist_ok=True)
        
        for sub_class in subclasses:
            # Vérification si la classe doit être ignorée (ex: 049)
            if any(sub_class.startswith(prefix) for prefix in IGNORE_PREFIXES):
                print(f"  -> Ignoré (règle manuelle) : {sub_class}")
                continue
            
            src_path = INPUT_DIR / sub_class
            
            # Vérifier si le dossier source existe
            if not src_path.exists():
                # print(f"  -> Attention : Dossier source introuvable : {sub_class}")
                continue

            # Traitement des fichiers
            files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
            
            for filename in files:
                original_file_path = src_path / filename
                
                # Construction du nouveau nom : SousClasse_NomFichier.jpg
                # new_filename = f"{sub_class}_{filename}" # désactivé, décommenter pour remettre
                new_filename = filename
                dest_file_path = dest_mother_path / new_filename
                
                try:
                    # Copie (shutil.copy2 préserve les métadonnées)
                    shutil.copy2(original_file_path, dest_file_path)
                    total_moved += 1
                except Exception as e:
                    print(f"Erreur copie {filename}: {e}")

    print(f"\n--- TERMINÉ ---")
    print(f"Total images réorganisées : {total_moved}")
    print(f"Les données sont prêtes dans : {OUTPUT_DIR}")

if __name__ == "__main__":
    reorganize_data()
