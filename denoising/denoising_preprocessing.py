"""
Real Denoising avec Restormer
Applique le denoising sur un dossier d'images
"""

import torch
import torch.nn.functional as F
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional, List


class RestormerDenoiser:
    """
    Wrapper pour Restormer Real Denoising
    """
    
    def __init__(
        self,
        weights_path: str = 'Denoising/pretrained_models/real_denoising.pth',
        device: Optional[str] = None
    ):
        """
        Args:
            weights_path: Chemin vers les poids du modèle
            device: 'cuda', 'cpu', ou None (auto-detect)
        """
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Paramètres du modèle pour Real Denoising
        self.parameters = {
            'inp_channels': 3,
            'out_channels': 3,
            'dim': 48,
            'num_blocks': [4, 6, 6, 8],
            'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8],
            'ffn_expansion_factor': 2.66,
            'bias': False,
            'LayerNorm_type': 'BiasFree',  # Important pour Real Denoising
            'dual_pixel_task': False
        }
        
        # Charger le modèle
        self.model = self._load_model(weights_path)
        self.img_multiple_of = 8
        
        print(f"Restormer chargé sur {self.device}")
    
    def _load_model(self, weights_path: str):
        """Charge le modèle Restormer"""
        # Charger l'architecture
        load_arch = run_path(
            os.path.join('/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi/denoising/Restormer/basicsr', 'models', 'archs', 'restormer_arch.py')
        )
        model = load_arch['Restormer'](**self.parameters)
        
        # Charger les poids
        checkpoint = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(checkpoint['params'])
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_image(self, filepath: str) -> np.ndarray:
        """Charge une image en RGB"""
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError(f"Impossible de charger {filepath}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _save_image(self, filepath: str, img: np.ndarray):
        """Sauvegarde une image"""
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """
        Prétraitement : numpy array → torch tensor
        Args:
            img: Array RGB [H, W, 3] en uint8 [0, 255]
        Returns:
            Tensor [1, 3, H, W] float32 [0, 1]
        """
        input_tensor = torch.from_numpy(img).float().div(255.)
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
        return input_tensor.to(self.device)
    
    def _pad_image(self, img_tensor: torch.Tensor) -> tuple:
        """
        Pad l'image pour être multiple de 8
        Returns:
            (padded_tensor, original_height, original_width)
        """
        height, width = img_tensor.shape[2], img_tensor.shape[3]
        
        H = ((height + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        W = ((width + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        
        padh = H - height if height % self.img_multiple_of != 0 else 0
        padw = W - width if width % self.img_multiple_of != 0 else 0
        
        padded = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')
        
        return padded, height, width
    
    def _postprocess(
        self,
        output_tensor: torch.Tensor,
        original_height: int,
        original_width: int
    ) -> np.ndarray:
        """
        Post-traitement : torch tensor → numpy array
        Args:
            output_tensor: [1, 3, H, W] float32 [0, 1]
        Returns:
            Array RGB [H, W, 3] uint8 [0, 255]
        """
        # Unpad
        output_tensor = output_tensor[:, :, :original_height, :original_width]
        
        # Clamp et convertir
        output_tensor = torch.clamp(output_tensor, 0, 1)
        output_np = output_tensor.permute(0, 2, 3, 1).cpu().detach().numpy()
        output_np = img_as_ubyte(output_np[0])
        
        return output_np
    
    @torch.no_grad()
    def denoise_image(
        self,
        input_path: str,
        output_path: str,
        tile_size: Optional[int] = None,
        tile_overlap: int = 32
    ):
        """
        Denoise une seule image
        
        Args:
            input_path: Chemin de l'image bruitée
            output_path: Chemin de sauvegarde
            tile_size: Si None, traite l'image entière. Sinon découpe en tiles
            tile_overlap: Chevauchement entre tiles (pixels)
        """
        # Libérer la mémoire GPU
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        
        # Charger et prétraiter
        img = self._load_image(input_path)
        input_tensor = self._preprocess(img)
        input_tensor, orig_h, orig_w = self._pad_image(input_tensor)
        
        # Inference
        if tile_size is None:
            # Traitement de l'image entière
            restored = self.model(input_tensor)
        else:
            # Traitement par tiles (pour grandes images)
            restored = self._denoise_with_tiles(input_tensor, tile_size, tile_overlap)
        
        # Post-traiter et sauvegarder
        restored_img = self._postprocess(restored, orig_h, orig_w)
        
        # Créer le dossier de sortie si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._save_image(output_path, restored_img)
    
    @torch.no_grad()
    def _denoise_with_tiles(
        self,
        input_tensor: torch.Tensor,
        tile_size: int,
        tile_overlap: int
    ) -> torch.Tensor:
        """
        Denoise par tiles pour grandes images
        """
        b, c, h, w = input_tensor.shape
        tile = min(tile_size, h, w)
        
        assert tile % 8 == 0, "tile_size doit être multiple de 8"
        
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        
        E = torch.zeros(b, c, h, w).type_as(input_tensor)
        W = torch.zeros_like(E)
        
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_tensor[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = self.model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)
                
                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
        
        restored = E.div_(W)
        return restored
    
    def denoise_folder(
        self,
        input_dir: str,
        output_dir: str,
        tile_size: Optional[int] = None,
        tile_overlap: int = 32,
        extensions: List[str] = None
    ):
        """
        Denoise tous les images d'un dossier
        
        Args:
            input_dir: Dossier contenant les images bruitées
            output_dir: Dossier de sortie
            tile_size: Taille des tiles (None = image entière)
            tile_overlap: Chevauchement entre tiles
            extensions: Liste des extensions à traiter
        """
        if extensions is None:
            extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        
        # Trouver tous les fichiers
        files = []
        for ext in extensions:
            files.extend(glob(os.path.join(input_dir, f'*.{ext}')))
        
        files = natsorted(files)
        
        if len(files) == 0:
            raise ValueError(f" Aucune image trouvée dans {input_dir}")
        
        print(f" {len(files)} images trouvées dans {input_dir}")
        print(f" Résultats dans {output_dir}\n")
        
        # Créer le dossier de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Traiter chaque image
        for file_path in tqdm(files, desc="Denoising"):
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)
            
            try:
                self.denoise_image(
                    input_path=file_path,
                    output_path=output_path,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap
                )
            except Exception as e:
                print(f"\n Erreur sur {filename}: {e}")
                continue
        
        print(f"\n Denoising terminé ! Résultats dans {output_dir}")


def main_train():
    """
    Denoise toutes les classes du dataset train
    """
    # Configuration des chemins
    BASE_INPUT_DIR = '/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi/data/train'
    BASE_OUTPUT_DIR = '/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi/denoising/preprocessed_data/train'
    WEIGHTS_PATH = '/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi/denoising/Restormer/Denoising/pretrained_models/real_denoising.pth'
    
    # Options de denoising
    USE_TILES = False  # True pour grandes images (> 2000x2000)
    TILE_SIZE = 720    # Taille des tiles si USE_TILES=True
    TILE_OVERLAP = 32  # Chevauchement entre tiles
    
    # Vérifier que le dossier source existe
    if not os.path.exists(BASE_INPUT_DIR):
        raise ValueError(f"❌ Le dossier {BASE_INPUT_DIR} n'existe pas !")
    
    # Lister tous les sous-dossiers (classes)
    class_folders = [
        d for d in os.listdir(BASE_INPUT_DIR) 
        if os.path.isdir(os.path.join(BASE_INPUT_DIR, d))
    ]
    class_folders = sorted(class_folders)  # Trier par ordre alphabétique
    
    if len(class_folders) == 0:
        raise ValueError(f"❌ Aucun sous-dossier trouvé dans {BASE_INPUT_DIR}")
    
    print(f" {len(class_folders)} classes trouvées")
    print(f" Structure : {BASE_INPUT_DIR}/[classe]/images.jpg")
    print(f" Sortie : {BASE_OUTPUT_DIR}/[classe]/images.jpg\n")
    
    # Initialiser le denoiser une seule fois
    print(" Initialisation du modèle Restormer...")
    denoiser = RestormerDenoiser(
        weights_path=WEIGHTS_PATH,
        device=None  # Auto-detect (cuda si disponible)
    )
    print()
    
    # Traiter chaque classe
    for idx, class_name in enumerate(class_folders, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(class_folders)}] Traitement de la classe : {class_name}")
        print(f"{'='*70}")
        
        # Chemins pour cette classe
        input_class_dir = os.path.join(BASE_INPUT_DIR, class_name)
        output_class_dir = os.path.join(BASE_OUTPUT_DIR, class_name)
        
        # Créer le dossier de sortie pour cette classe
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Compter les images
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        image_files = []
        for ext in extensions:
            image_files.extend(glob(os.path.join(input_class_dir, f'*.{ext}')))
        
        num_images = len(image_files)
        
        if num_images == 0:
            print(f"  Aucune image trouvée dans {class_name}, passage à la suivante")
            continue
        
        print(f" {num_images} images à traiter")
        
        try:
            # Denoise tous les images de cette classe
            denoiser.denoise_folder(
                input_dir=input_class_dir,
                output_dir=output_class_dir,
                tile_size=TILE_SIZE if USE_TILES else None,
                tile_overlap=TILE_OVERLAP
            )
            print(f" Classe {class_name} terminée !")
            
        except Exception as e:
            print(f" Erreur lors du traitement de {class_name}: {e}")
            print(f" Passage à la classe suivante...")
            continue
    
    print(f"\n{'='*70}")
    print(f" DENOISING TERMINÉ !")
    print(f"{'='*70}")
    print(f" Données nettoyées disponibles dans :")
    print(f"   {BASE_OUTPUT_DIR}")
    print(f"\n Résumé :")
    print(f"   - Classes traitées : {len(class_folders)}")
    
    # Compter le nombre total d'images traitées
    total_images = 0
    for class_name in class_folders:
        output_class_dir = os.path.join(BASE_OUTPUT_DIR, class_name)
        if os.path.exists(output_class_dir):
            class_images = []
            for ext in extensions:
                class_images.extend(glob(os.path.join(output_class_dir, f'*.{ext}')))
            total_images += len(class_images)
    
    print(f" - Images traitées : {total_images}")

def main_test():
    """
    Denoise toutes les images du dataset test (pas de sous-dossiers)
    """
    # Configuration des chemins
    INPUT_DIR = '/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi/data/test/imgs'
    OUTPUT_DIR = '/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi/denoising/preprocessed_data/test'
    WEIGHTS_PATH = '/home/lmalerba/projects/sdim_data_challenge/kaggle_challenge_sdi/denoising/Restormer/Denoising/pretrained_models/real_denoising.pth'
    
    # Options de denoising
    USE_TILES = False  # True pour grandes images (> 2000x2000)
    TILE_SIZE = 720    # Taille des tiles si USE_TILES=True
    TILE_OVERLAP = 32  # Chevauchement entre tiles
    
    # Vérifier que le dossier source existe
    if not os.path.exists(INPUT_DIR):
        raise ValueError(f"❌ Le dossier {INPUT_DIR} n'existe pas !")
    
    # Compter les images dans le dossier test
    extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(INPUT_DIR, f'*.{ext}')))
    
    num_images = len(image_files)
    
    if num_images == 0:
        raise ValueError(f"❌ Aucune image trouvée dans {INPUT_DIR}")
    
    print(f"📊 {num_images} images de test trouvées")
    print(f"📁 Source : {INPUT_DIR}")
    print(f"💾 Sortie : {OUTPUT_DIR}\n")
    
    # Créer le dossier de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialiser le denoiser
    print("🔧 Initialisation du modèle Restormer...")
    denoiser = RestormerDenoiser(
        weights_path=WEIGHTS_PATH,
        device=None  # Auto-detect (cuda si disponible)
    )
    print()
    
    print(f"{'='*70}")
    print(f"🚀 DÉMARRAGE DU DENOISING DES IMAGES DE TEST")
    print(f"{'='*70}\n")
    
    try:
        # Denoise toutes les images du dossier test
        denoiser.denoise_folder(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            tile_size=TILE_SIZE if USE_TILES else None,
            tile_overlap=TILE_OVERLAP
        )
        
        print(f"\n{'='*70}")
        print(f"✅ DENOISING TERMINÉ !")
        print(f"{'='*70}")
        print(f"💾 Données nettoyées disponibles dans :")
        print(f"   {OUTPUT_DIR}")
        
        # Compter le nombre d'images traitées
        output_images = []
        for ext in extensions:
            output_images.extend(glob(os.path.join(OUTPUT_DIR, f'*.{ext}')))
        
        print(f"\n📊 Résumé :")
        print(f"   - Images traitées : {len(output_images)}/{num_images}")
        
    except Exception as e:
        print(f"\n❌ Erreur lors du traitement : {e}")
        raise

if __name__ == '__main__':
    main_test()

