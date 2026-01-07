import os
import sys
import yaml
import torch
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import math
import numpy as np # Parfois nécessaire pour data.py

# Ajout du path courant
sys.path.append(os.getcwd())

# ---------------------------------------------------------
# IMPORTATIONS PERSONNALISÉES
# ---------------------------------------------------------

# 1. On essaye d'importer data correctement
try:
    import data # Ton fichier data.py doit être à côté
    # On vérifie que la fonction existe bien
    if not hasattr(data, 'get_transforms'):
        raise ImportError("La fonction 'get_transforms' est introuvable dans data.py")
except ImportError as e:
    print(f"❌ ERREUR CRITIQUE : {e}")
    sys.exit(1)

try:
    from torchtmpl.models import build_model
except ImportError:
    print("⚠️ Attention : Impossible d'importer 'torchtmpl'. Assure-toi que build_model est dispo.")

# =========================================================================
# ⚙️ CONFIGURATION
# =========================================================================

# Dossiers globaux
TEST_IMAGES_DIR = "./data/original_dataset/test"
OUTPUT_CSV = "./results/submission_hierarchical.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
NUM_WORKERS = 8

# A. CONFIGURATION GÉNÉRALISTE
GENERALIST = {
    "weights": "./logs_v2/reannotated/DINOv3Classifier_bystep_3/best_model.pt",
    "config":  "./logs_v2/reannotated/DINOv3Classifier_bystep_3/config.yaml",
}

# B. CONFIGURATION SPÉCIALISTES
SPECIALIST_TEMPLATE_YAML = "config_orchestrator_1.yaml" 
TRAIN_DATA_ROOT = "./data/data_reannotated_v2/train" 

SPECIALISTS_WEIGHTS = {
    "1_Copepoda_Torpille":      "./logs_v2/specialists_1/1_Copepoda_Torpille/DINOv3Classifier_bystep_0/best_model.pt",
    "2_Gelatinous_Transparent": "./logs_v2/specialists_1/2_Gelatinous_Transparent/DINOv3Classifier_bystep_0/best_model.pt",
    "3_Macro_Crustacea_Larvae": "./logs_v2/specialists_1/3_Macro_Crustacea_Larvae/DINOv3Classifier_bystep_0/best_model.pt",
    "4_Spherical_Compact_Eggs": "./logs_v2/specialists_1/4_Spherical_Compact_Eggs/DINOv3Classifier_bystep_0/best_model.pt",
    "5_Elongated_and_Chains":   "./logs_v2/specialists_1/5_Elongated_and_Chains/DINOv3Classifier_bystep_0/best_model.pt",
    "6_Hard_Shells_Misc":       "./logs_v2/specialists_1/6_Hard_Shells_Misc/DINOv3Classifier_bystep_0/best_model.pt",
    "7_Detritus_Noise_Empty":   "./logs_v2/specialists_1/7_Detritus_Noise_Empty/DINOv3Classifier_bystep_0/best_model.pt",
    "8_Triangulaire_Allonge":   "./logs_v2/specialists_1/8_Triangulaire_Allonge/DINOv3Classifier_bystep_0/best_model.pt"
}

# =========================================================================
# UTILITAIRES INFÉRENCE
# =========================================================================

class InferenceDataset(Dataset):
    """
    Dataset simple qui lit une liste de fichiers.
    N'utilise PAS la structure de dossier (ImageFolder) car le dossier Test est plat ou inconnu.
    """
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            # On force le convert RGB pour uniformiser
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error opening {path}: {e}")
            image = Image.new('RGB', (224, 224)) # Image noire de fallback

        if self.transform:
            image = self.transform(image)

        return image, os.path.basename(path)


def run_inference(model, images_list, transform_pipeline, desc="Inference"):
    """
    Lance l'inférence.
    IMPORTANT : On passe 'transform_pipeline' (l'objet complet) et pas juste une taille.
    """
    # Création du dataset avec les BONNES transformations (issues de data.py)
    dataset = InferenceDataset(images_list, transform=transform_pipeline)
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    predictions = {}
    model.eval()

    with torch.no_grad():
        for imgs, fnames in tqdm(loader, desc=desc):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            for f, p in zip(fnames, preds.cpu().numpy()):
                predictions[f] = p
    return predictions


# =========================================================================
# MAIN LOOP
# =========================================================================

def main():
    print("🚀 Démarrage de l'inférence hiérarchique (Sync with Training)")

    if not os.path.exists(TRAIN_DATA_ROOT):
        print(f"❌ ERREUR : Dossier racine d'entraînement introuvable : {TRAIN_DATA_ROOT}")
        return

    # 1. SCAN IMAGES DE TEST
    all_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "**", "*.*"), recursive=True)
    valid_exts = {'.jpg', '.png', '.jpeg', '.tif', '.bmp'}
    all_images = [x for x in all_images if os.path.splitext(x)[1].lower() in valid_exts]
    print(f"📂 Images à traiter : {len(all_images)}")

    if not all_images:
        print("Aucune image de test trouvée.")
        return

    # ---------------------------------------------------------
    # 2. GÉNÉRALISTE
    # ---------------------------------------------------------
    print("\n========= 1. GENERALIST PREDICTION =========")
    
    # Chargement Config
    with open(GENERALIST['config'], 'r') as f:
        gen_conf = yaml.safe_load(f)
    
    # Récupération des paramètres pour reconstruire la transfo EXACTE
    gen_data_conf = gen_conf['data']
    gen_resize    = gen_data_conf.get('images_resize', 224)
    gen_version   = gen_data_conf.get('transformations_version', 'basic')
    gen_use_rgb   = gen_data_conf.get('use_rgb', True)

    print(f"   ⚙️ Config Généraliste : Resize={gen_resize}, Algo='{gen_version}'")

    # --- CORRECTION MAJEURE : Utilisation de data.get_transforms ---
    gen_transform_pipeline = data.get_transforms(
        version=gen_version,
        train=False,        # IMPORTANT : Mode validation/test (pas d'augmentation)
        use_rgb=gen_use_rgb,
        resize=gen_resize
    )
    # ---------------------------------------------------------------

    # Récupération de l'ordre des classes via data.get_dataloaders (juste pour la méta-data)
    # On crée une config temporaire pour tromper get_dataloaders
    temp_gen_config = gen_data_conf.copy()
    temp_gen_config["trainpath"] = TRAIN_DATA_ROOT
    
    # On coupe la verbosité de get_dataloaders si possible ou on ignore
    try:
        _, _, _, gen_num_classes, _, gen_classes_ordered = data.get_dataloaders(
            temp_gen_config, 
            training_style=temp_gen_config.get("training_style", "standard")
        )
    except Exception as e:
        print(f"❌ ERREUR Chargement structure Généraliste : {e}")
        return

    print(f"   ✅ Classes ({gen_num_classes}) : {gen_classes_ordered}")

    # Chargement Modèle
    model_gen = build_model(gen_conf['model'], gen_resize, gen_num_classes)
    model_gen.load_state_dict(torch.load(GENERALIST['weights'], map_location=DEVICE))
    model_gen.to(DEVICE)

    # Inférence Généraliste
    gen_preds_raw = run_inference(model_gen, all_images, gen_transform_pipeline, desc="Generalist")

    del model_gen
    torch.cuda.empty_cache()

    # Dispatching
    dispatcher = {c: [] for c in gen_classes_ordered} 
    fname_to_path = {os.path.basename(p): p for p in all_images}
    final_results = {}

    for fname, idx in gen_preds_raw.items():
        if 0 <= idx < len(gen_classes_ordered):
            dispatcher[gen_classes_ordered[idx]].append(fname_to_path[fname])
        else:
            final_results[fname] = "UNKNOWN"

    # ---------------------------------------------------------
    # 3. SPÉCIALISTES
    # ---------------------------------------------------------
    print("\n========= 2. SPECIALIST PREDICTION =========")

    # Chargement du Template YAML des spécialistes
    with open(SPECIALIST_TEMPLATE_YAML, 'r') as f:
        template_conf = yaml.safe_load(f)
    
    spec_data_conf_tmpl = template_conf['data']
    
    # Paramètres de transformation du Template (supposés identiques pour tous les spécialistes)
    spec_resize    = spec_data_conf_tmpl.get('images_resize', 224)
    spec_version   = spec_data_conf_tmpl.get('transformations_version', 'basic')
    spec_use_rgb   = spec_data_conf_tmpl.get('use_rgb', True)
    
    print(f"   ⚙️ Config Spécialistes (Template) : Resize={spec_resize}, Algo='{spec_version}'")
    
    # --- Création du Transform Pipeline Spécialiste ---
    spec_transform_pipeline = data.get_transforms(
        version=spec_version,
        train=False,
        use_rgb=spec_use_rgb,
        resize=spec_resize
    )

    MAX_IMAGES_PER_TRAIN = 150000 

    for gen_class_name, img_paths in dispatcher.items():
        if not img_paths:
            continue

        print(f"\n🔵 Classe Mère : {gen_class_name} ({len(img_paths)} images)")

        # Checks de base
        if gen_class_name not in SPECIALISTS_WEIGHTS:
            print(f"   -> Pas de spécialiste. Label = {gen_class_name}")
            for p in img_paths: final_results[os.path.basename(p)] = gen_class_name
            continue
        
        weights_path = SPECIALISTS_WEIGHTS[gen_class_name]
        if not os.path.exists(weights_path):
             print(f"   -> Poids introuvables. Fallback.")
             for p in img_paths: final_results[os.path.basename(p)] = gen_class_name
             continue

        # --- A. Chargement Structure des classes (Divider, Loader) ---
        current_train_dir = os.path.join(TRAIN_DATA_ROOT, gen_class_name)
        
        # Calcul Divider (reproduction logique Training)
        num_files_train = 0
        if os.path.exists(current_train_dir):
            num_files_train = sum([len(files) for r, d, files in os.walk(current_train_dir)])
        
        divider = 1
        if num_files_train > MAX_IMAGES_PER_TRAIN:
            divider = math.ceil(num_files_train / MAX_IMAGES_PER_TRAIN)

        # Config temporaire pour data.get_dataloaders
        this_spec_conf = spec_data_conf_tmpl.copy()
        this_spec_conf["trainpath"] = current_train_dir
        this_spec_conf["dataset_size_divider"] = divider

        try:
            # On demande à data.py de scanner le dossier pour nous donner l'ordre des classes
            _, _, _, sub_num_classes, _, sub_classes_ordered = data.get_dataloaders(
                this_spec_conf, 
                training_style="specialist"
            )
        except Exception as e:
            print(f"   ❌ Erreur lecture classes : {e}")
            for p in img_paths: final_results[os.path.basename(p)] = gen_class_name
            continue

        if not sub_classes_ordered:
            print(f"   ❌ Aucune sous-classe. Fallback.")
            for p in img_paths: final_results[os.path.basename(p)] = gen_class_name
            continue

        # --- B. Validation Modèle ---
        try:
            ckpt = torch.load(weights_path, map_location=DEVICE)
            # Vérification basique des dimensions
            sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            last_keys = [k for k in sd.keys() if 'weight' in k and ('fc' in k or 'head' in k or 'classifier' in k)]
            if last_keys:
                num_ckpt = sd[last_keys[-1]].shape[0]
                if num_ckpt != sub_num_classes:
                    print(f"   ❌ mismatch: Dossier={sub_num_classes} vs Modèle={num_ckpt}. Fallback.")
                    for p in img_paths: final_results[os.path.basename(p)] = gen_class_name
                    del ckpt
                    continue
        except Exception:
            pass # On tente quand même si on n'arrive pas à vérifier

        # --- C. Inférence ---
        # Note : On réutilise spec_transform_pipeline calculé plus haut
        spec_channels = 3 if spec_use_rgb else 1
        spec_input_shape = (spec_channels, spec_resize, spec_resize)

        # On passe spec_input_shape (tuple) au lieu de spec_resize (int)
        model_spec = build_model(template_conf['model'], spec_input_shape, sub_num_classes)
        model_spec.load_state_dict(ckpt)
        model_spec.to(DEVICE)
        
        spec_preds = run_inference(model_spec, img_paths, spec_transform_pipeline, desc=f"   -> {gen_class_name}")
        for fname, idx in spec_preds.items():
            if 0 <= idx < len(sub_classes_ordered):
                final_results[fname] = sub_classes_ordered[idx]
            else:
                final_results[fname] = gen_class_name

        del model_spec, ckpt
        torch.cuda.empty_cache()

    # 4. SAUVEGARDE
    print("\n========= 3. SAVING =========")
    df = pd.DataFrame(list(final_results.items()), columns=['imgname', 'label'])
    df = df.sort_values(by='imgname')
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Résultats sauvegardés : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
