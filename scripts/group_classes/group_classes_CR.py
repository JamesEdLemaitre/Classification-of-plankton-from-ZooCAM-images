import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import math

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = Path('/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi/data_reannotated/train')
OUTPUT_DIR = Path('/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi/group_classes/visualisation_final_check')
SAMPLES_PER_CLASS = 16  # Nombre d'images à afficher (4x4)

def visualize_classes():
    # Création du dossier de sortie
    if not OUTPUT_DIR.exists():
        os.makedirs(OUTPUT_DIR)
    
    # Récupérer la liste des sous-dossiers (les classes mères)
    if not INPUT_DIR.exists():
        print(f"Erreur : Le dossier {INPUT_DIR} n'existe pas. Lance le script de réorganisation d'abord.")
        return

    classes = [d for d in os.listdir(INPUT_DIR) if (INPUT_DIR / d).is_dir()]
    classes.sort() # Pour avoir un ordre propre

    print(f"Génération des visualisations pour {len(classes)} classes mères...")

    for class_name in classes:
        class_path = INPUT_DIR / class_name
        
        # Récupérer toutes les images du dossier
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        
        # S'il n'y a pas d'images, on passe
        if not images:
            print(f"  -> {class_name} : Dossier vide (ignoré).")
            continue

        # Sélection aléatoire de 16 images (ou moins si le dossier en contient moins)
        nb_samples = min(len(images), SAMPLES_PER_CLASS)
        selected_images = random.sample(images, nb_samples)
        
        # Création de la figure (Grille 4x4)
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle(f"Classe Mère : {class_name} ({len(images)} images totales)", fontsize=20, weight='bold')
        
        # Aplatir les axes pour itérer facilement (tableau 1D au lieu de 2D)
        axes_flat = axes.flatten()

        for i, ax in enumerate(axes_flat):
            if i < nb_samples:
                img_name = selected_images[i]
                img_path = class_path / img_name
                
                try:
                    img = mpimg.imread(str(img_path))
                    ax.imshow(img)
                    
                    # Titre avec le nom du fichier (sans le préfixe de classe trop long si possible)
                    # On essaie de garder juste le nom original pour lisibilité
                    display_name = img_name
                    if len(display_name) > 25:
                        display_name = "..." + display_name[-25:]
                    
                    ax.set_title(display_name, fontsize=8)
                except Exception as e:
                    ax.text(0.5, 0.5, "Erreur lecture", ha='center', va='center')
            
            # Enlever les axes (chiffres x/y) pour la clarté
            ax.axis('off')

        # Ajustement de l'espace
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Sauvegarde
        save_path = OUTPUT_DIR / f"Check_{class_name}.png"
        plt.savefig(save_path)
        plt.close(fig) # Fermer pour libérer la mémoire
        
        print(f"  -> Sauvegardé : {save_path}")

    print("\n--- VISUALISATION TERMINÉE ---")
    print(f"Ouvre le dossier '{OUTPUT_DIR}' pour vérifier tes classes.")

if __name__ == "__main__":
    visualize_classes()
