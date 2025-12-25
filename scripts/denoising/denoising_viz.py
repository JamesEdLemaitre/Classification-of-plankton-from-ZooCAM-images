"""
Visualisation comparaison avant/après denoising
Crée des images side-by-side pour chaque classe
Gère les classes non encore preprocessées
"""

import os
import cv2
import numpy as np
from pathlib import Path
from glob import glob
import random
from tqdm import tqdm
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sans interface graphique


class DenoisingVisualizer:
    """
    Crée des visualisations avant/après denoising
    """
    
    def __init__(
        self,
        original_dir: str,
        denoised_dir: str,
        output_dir: str,
        images_per_class: int = 1
    ):
        """
        Args:
            original_dir: Dossier avec images originales (ex: data/train)
            denoised_dir: Dossier avec images denoised (ex: preprocessed_data/train)
            output_dir: Dossier pour sauver les comparaisons
            images_per_class: Nombre d'images à visualiser par classe
        """
        self.original_dir = Path(original_dir)
        self.denoised_dir = Path(denoised_dir)
        self.output_dir = Path(output_dir)
        self.images_per_class = images_per_class
        
        # Créer le dossier de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Original : {self.original_dir}")
        print(f"Denoised : {self.denoised_dir}")
        print(f"Sortie : {self.output_dir}")
    
    def _load_image(self, filepath: Path) -> Optional[np.ndarray]:
        """Charge une image en RGB"""
        try:
            img = cv2.imread(str(filepath))
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Erreur chargement {filepath}: {e}")
            return None
    
    def _get_available_classes(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Identifie les classes disponibles
        
        Returns:
            (classes_with_pairs, classes_original_only, classes_denoised_only)
        """
        # Classes dans original
        original_classes = set([
            d.name for d in self.original_dir.iterdir() 
            if d.is_dir()
        ])
        
        # Classes dans denoised
        if self.denoised_dir.exists():
            denoised_classes = set([
                d.name for d in self.denoised_dir.iterdir() 
                if d.is_dir()
            ])
        else:
            denoised_classes = set()
        
        # Classes avec paires (présentes dans les deux)
        classes_with_pairs = sorted(original_classes & denoised_classes)
        
        # Classes seulement dans original (pas encore preprocessées)
        classes_original_only = sorted(original_classes - denoised_classes)
        
        # Classes seulement dans denoised (orphelines)
        classes_denoised_only = sorted(denoised_classes - original_classes)
        
        return classes_with_pairs, classes_original_only, classes_denoised_only
    
    def _get_matching_images(
        self,
        class_name: str
    ) -> List[Tuple[Path, Path]]:
        """
        Trouve les paires d'images (original, denoised) pour une classe
        
        Returns:
            Liste de tuples (chemin_original, chemin_denoised)
        """
        original_class_dir = self.original_dir / class_name
        denoised_class_dir = self.denoised_dir / class_name
        
        # Vérifier que les dossiers existent
        if not original_class_dir.exists() or not denoised_class_dir.exists():
            return []
        
        # Extensions à chercher
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        
        # Lister les images originales
        original_images = []
        for ext in extensions:
            original_images.extend(original_class_dir.glob(f'*.{ext}'))
        
        if len(original_images) == 0:
            return []
        
        # Lister les images denoised (TOUTES extensions possibles)
        denoised_images = []
        for ext in extensions:
            denoised_images.extend(denoised_class_dir.glob(f'*.{ext}'))
        
        if len(denoised_images) == 0:
            return []
        
        # Créer un dictionnaire {stem: path} pour denoised
        denoised_dict = {img.stem: img for img in denoised_images}
        
        # Trouver les paires correspondantes
        pairs = []
        for orig_path in original_images:
            if orig_path.stem in denoised_dict:
                pairs.append((orig_path, denoised_dict[orig_path.stem]))
        
        return pairs

    
    def _create_comparison(
        self,
        img_original: np.ndarray,
        img_denoised: np.ndarray,
        class_name: str,
        image_name: str
    ) -> np.ndarray:
        """
        Crée une image de comparaison side-by-side avec matplotlib
        
        Returns:
            Image de comparaison (numpy array RGB)
        """
        # Créer la figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Classe: {class_name} | Image: {image_name}', 
                     fontsize=16, fontweight='bold')
        
        # Image originale
        axes[0].imshow(img_original)
        axes[0].set_title('🔴 Original (Noisy)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Image denoised
        axes[1].imshow(img_denoised)
        axes[1].set_title('✅ Denoised (Restormer)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Ajouter des statistiques
        orig_mean = img_original.mean()
        orig_std = img_original.std()
        den_mean = img_denoised.mean()
        den_std = img_denoised.std()
        
        stats_text = (
            f'Stats Original: μ={orig_mean:.1f}, σ={orig_std:.1f}\n'
            f'Stats Denoised: μ={den_mean:.1f}, σ={den_std:.1f}'
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                 family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Convertir la figure en numpy array
        fig.canvas.draw()
        img_comparison = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_comparison = img_comparison.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return img_comparison
    
    def _create_simple_comparison(
        self,
        img_original: np.ndarray,
        img_denoised: np.ndarray
    ) -> np.ndarray:
        """
        Crée une comparaison simple (concat horizontale)
        Plus rapide que matplotlib
        """
        # Redimensionner si nécessaire pour que les hauteurs matchent
        h1, w1 = img_original.shape[:2]
        h2, w2 = img_denoised.shape[:2]
        
        if h1 != h2:
            # Redimensionner la denoised pour matcher l'original
            img_denoised = cv2.resize(img_denoised, (w2, h1))
        
        # Ajouter du texte
        img_orig_copy = img_original.copy()
        img_den_copy = img_denoised.copy()
        
        # Texte "Original"
        cv2.putText(img_orig_copy, 'ORIGINAL', (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        
        # Texte "Denoised"
        cv2.putText(img_den_copy, 'DENOISED', (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Ligne de séparation
        separator = np.ones((h1, 5, 3), dtype=np.uint8) * 255
        
        # Concaténer horizontalement
        comparison = np.hstack([img_orig_copy, separator, img_den_copy])
        
        return comparison
    
    def visualize_all_classes(
        self,
        use_matplotlib: bool = True,
        random_selection: bool = True
    ):
        """
        Crée les visualisations pour toutes les classes disponibles
        
        Args:
            use_matplotlib: Si True, utilise matplotlib (plus beau mais plus lent)
            random_selection: Si True, sélectionne aléatoirement, sinon prend les premières
        """
        # Identifier les classes disponibles
        classes_with_pairs, classes_original_only, classes_denoised_only = self._get_available_classes()
        
        # Afficher le résumé
        print(f"\n{'='*70}")
        print(f"ANALYSE DES CLASSES")
        print(f"{'='*70}")
        print(f"Classes avec paires (original + denoised): {len(classes_with_pairs)}")
        print(f"Classes non preprocessées (original seulement): {len(classes_original_only)}")
        print(f"Classes orphelines (denoised seulement): {len(classes_denoised_only)}")
        
        if len(classes_original_only) > 0:
            print(f"\n⏳ Classes en attente de preprocessing:")
            for i, cls in enumerate(classes_original_only[:10], 1):  # Afficher max 10
                print(f"   {i}. {cls}")
            if len(classes_original_only) > 10:
                print(f"   ... et {len(classes_original_only) - 10} autres")
        
        if len(classes_with_pairs) == 0:
            print(f"\n❌ Aucune classe avec paires trouvée !")
            print(f"💡 Lancez d'abord le preprocessing avec denoise_dataset.py")
            return
        
        print(f"\nSélection: {'Aléatoire' if random_selection else 'Première(s)'}")
        print(f"Style: {'Matplotlib (qualité)' if use_matplotlib else 'OpenCV (rapide)'}\n")
        
        total_comparisons = 0
        classes_processed = 0
        classes_with_errors = []
        
        # Traiter uniquement les classes avec paires
        for idx, class_name in enumerate(tqdm(classes_with_pairs, desc="Classes"), 1):
            
            # Trouver les paires d'images
            pairs = self._get_matching_images(class_name)
            
            if len(pairs) == 0:
                classes_with_errors.append(class_name)
                continue
            
            # Sélectionner les images à visualiser
            num_to_select = min(self.images_per_class, len(pairs))
            
            if random_selection:
                selected_pairs = random.sample(pairs, num_to_select)
            else:
                selected_pairs = pairs[:num_to_select]
            
            # Créer les comparaisons
            for pair_idx, (orig_path, den_path) in enumerate(selected_pairs, 1):
                
                # Charger les images
                img_original = self._load_image(orig_path)
                img_denoised = self._load_image(den_path)
                
                if img_original is None or img_denoised is None:
                    continue
                
                # Créer la comparaison
                try:
                    if use_matplotlib:
                        comparison = self._create_comparison(
                            img_original, img_denoised,
                            class_name, orig_path.name
                        )
                    else:
                        comparison = self._create_simple_comparison(
                            img_original, img_denoised
                        )
                    
                    # Sauvegarder
                    if self.images_per_class == 1:
                        output_filename = f"{class_name}.png"
                    else:
                        output_filename = f"{class_name}_{pair_idx:02d}.png"
                    
                    output_path = self.output_dir / output_filename
                    
                    # Sauver (convertir RGB -> BGR pour OpenCV)
                    cv2.imwrite(
                        str(output_path),
                        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                    )
                    
                    total_comparisons += 1
                
                except Exception as e:
                    print(f"⚠️  Erreur lors de la création de la comparaison pour {class_name}: {e}")
                    classes_with_errors.append(class_name)
                    continue
            
            classes_processed += 1
        
        # Afficher le résumé final
        print(f"\n{'='*70}")
        print(f"VISUALISATIONS TERMINÉES !")
        print(f"{'='*70}")
        print(f"Statistiques:")
        print(f"   - Classes avec paires disponibles: {len(classes_with_pairs)}")
        print(f"   - Classes traitées avec succès: {classes_processed}")
        print(f"   - Comparaisons créées: {total_comparisons}")
        print(f"   - Classes en attente de preprocessing: {len(classes_original_only)}")
        
        if len(classes_with_errors) > 0:
            print(f"\n  Classes avec erreurs ({len(classes_with_errors)}):")
            for cls in classes_with_errors[:5]:
                print(f"   - {cls}")
            if len(classes_with_errors) > 5:
                print(f"   ... et {len(classes_with_errors) - 5} autres")
        
        print(f"\n Dossier de sortie: {self.output_dir}")
        
    
    def create_summary_grid(
        self,
        grid_size: Tuple[int, int] = (4, 4),
        output_filename: str = 'summary_grid.png'
    ):
        """
        Crée une grille de toutes les comparaisons (uniquement classes avec paires)
        
        Args:
            grid_size: (rows, cols) de la grille
            output_filename: Nom du fichier de sortie
        """
        rows, cols = grid_size
        num_samples = rows * cols
        
        # Identifier les classes avec paires
        classes_with_pairs, _, _ = self._get_available_classes()
        
        if len(classes_with_pairs) == 0:
            print("Aucune classe avec paires trouvée pour la grille !")
            return
        
        # Sélectionner aléatoirement des classes
        selected_classes = random.sample(
            classes_with_pairs, 
            min(num_samples, len(classes_with_pairs))
        )
        
        print(f"Création d'une grille {rows}x{cols} avec {len(selected_classes)} classes...")
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axes = axes.flatten() if rows * cols > 1 else [axes]
        
        valid_comparisons = 0
        
        for idx, class_name in enumerate(tqdm(selected_classes, desc="Grille")):
            if idx >= len(axes):
                break
            
            # Trouver une paire d'images
            pairs = self._get_matching_images(class_name)
            if len(pairs) == 0:
                axes[idx].axis('off')
                axes[idx].text(0.5, 0.5, f'{class_name}\n(no pairs)', 
                              ha='center', va='center', fontsize=8)
                continue
            
            orig_path, den_path = random.choice(pairs)
            
            # Charger les images
            img_original = self._load_image(orig_path)
            img_denoised = self._load_image(den_path)
            
            if img_original is None or img_denoised is None:
                axes[idx].axis('off')
                axes[idx].text(0.5, 0.5, f'{class_name}\n(load error)', 
                              ha='center', va='center', fontsize=8)
                continue
            
            try:
                # Créer une mini-comparaison
                h, w = img_original.shape[:2]
                img_denoised_resized = cv2.resize(img_denoised, (w, h))
                
                # Concat horizontal
                comparison = np.hstack([img_original, img_denoised_resized])
                
                # Afficher
                axes[idx].imshow(comparison)
                axes[idx].set_title(class_name, fontsize=10, fontweight='bold')
                axes[idx].axis('off')
                
                valid_comparisons += 1
                
            except Exception as e:
                axes[idx].axis('off')
                axes[idx].text(0.5, 0.5, f'{class_name}\n(error)', 
                              ha='center', va='center', fontsize=8)
                print(f" Erreur grille pour {class_name}: {e}")
        
        # Masquer les axes vides
        for idx in range(len(selected_classes), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Comparaison Denoising: Original (gauche) vs Denoised (droite)\n'
                    f'{valid_comparisons}/{len(selected_classes)} comparaisons valides',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Sauvegarder
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Grille sauvegardée: {output_path}")
        print(f"   ({valid_comparisons} comparaisons valides sur {len(selected_classes)})")


def main():
    """
    Crée des visualisations de comparaison avant/après denoising
    """
    # Configuration
    ORIGINAL_DIR = '/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi/data/train'
    DENOISED_DIR = '/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi/denoising/preprocessed_data/train'
    OUTPUT_DIR = '/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi/denoising/visualizations'
    
    # Options
    IMAGES_PER_CLASS = 1  # Nombre d'images à visualiser par classe
    USE_MATPLOTLIB = True  # True = belle qualité, False = rapide
    RANDOM_SELECTION = True  # True = aléatoire, False = premières images
    CREATE_GRID = True  # Créer aussi une grille récapitulative
    
    # Créer le visualizer
    visualizer = DenoisingVisualizer(
        original_dir=ORIGINAL_DIR,
        denoised_dir=DENOISED_DIR,
        output_dir=OUTPUT_DIR,
        images_per_class=IMAGES_PER_CLASS
    )
    
    # Créer les comparaisons individuelles
    print("\n" + "="*70)
    print("CRÉATION DES COMPARAISONS INDIVIDUELLES")
    print("="*70)
    
    visualizer.visualize_all_classes(
        use_matplotlib=USE_MATPLOTLIB,
        random_selection=RANDOM_SELECTION
    )
    
    # Créer une grille récapitulative (optionnel)
    if CREATE_GRID:
        print("\n" + "="*70)
        print("CRÉATION DE LA GRILLE RÉCAPITULATIVE")
        print("="*70)
        
        visualizer.create_summary_grid(
            grid_size=(6, 6),  # Grille 6x6 = 36 classes
            output_filename='summary_grid_6x6.png'
        )
    
    print("\n" + "="*70)
    print("🎉 TOUT EST TERMINÉ !")
    print("="*70)
    print(f"\n Visualisations disponibles dans:")
    print(f"   {OUTPUT_DIR}\n")


if __name__ == '__main__':
    main()
