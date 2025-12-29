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

# Ajout du path courant
sys.path.append(os.getcwd())

# ---------------------------------------------------------
# IMPORTATIONS PERSONNALISÉES
# ---------------------------------------------------------
try:
    from torchtmpl.models import build_model
except ImportError:
    print("⚠️ Attention : Impossible d'importer 'torchtmpl'.")

try:
    # On importe votre classe dataset pour récupérer la logique de nommage (split('_')[0])
    from data import SpecialistDataset
except ImportError:
    print("❌ ERREUR CRITIQUE : Impossible d'importer 'SpecialistDataset' depuis 'data.py'.")
    print("   Vérifie que le fichier data.py est bien à la racine.")
    sys.exit(1)

# =========================================================================
# ⚙️ CONFIGURATION
# =========================================================================

# Dossiers globaux
TEST_IMAGES_DIR = "./data/original_dataset/test"
OUTPUT_CSV = "./results/submission_hierarchical.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 4

# A. CONFIGURATION GÉNÉRALISTE
GENERALIST = {
    "weights": "./logs_v2/reannotated/DINOv3Classifier_bystep_1/best_model.pt",
    "config":  "./logs_v2/reannotated/DINOv3Classifier_bystep_1/config.yaml",
    "classes_order": [
        "1_Copepoda_Torpille",
        "2_Gelatinous_Transparent",
        "3_Macro_Crustacea_Larvae",
        "4_Spherical_Compact_Eggs",
        "5_Elongated_and_Chains",
        "6_Hard_Shells_Misc",
        "7_Detritus_Noise_Empty",
        "8_Triangulaire_Allonge"
    ]
}

# B. CONFIGURATION SPÉCIALISTES
SPECIALIST_TEMPLATE_YAML = "config_orchestrator_0.yaml" 
TRAIN_DATA_ROOT = "./data/data_reannotated/train" # Dossier contenant 1_Copepoda, 2_Gela..., etc.

SPECIALISTS_WEIGHTS = {
    "1_Copepoda_Torpille":      "./logs_v2/specialists/1_Copepoda_Torpille/DINOv3Classifier_bystep_7/best_model.pt",
    "2_Gelatinous_Transparent": "./logs_v2/specialists/2_Gelatinous_Transparent/DINOv3Classifier_bystep_7/best_model.pt",
    "3_Macro_Crustacea_Larvae": "./logs_v2/specialists/3_Macro_Crustacea_Larvae/DINOv3Classifier_bystep_2/best_model.pt",
    "4_Spherical_Compact_Eggs": "./logs_v2/specialists/4_Spherical_Compact_Eggs/DINOv3Classifier_bystep_2/best_model.pt",
    "5_Elongated_and_Chains":   "./logs_v2/specialists/5_Elongated_and_Chains/DINOv3Classifier_bystep_2/best_model.pt",
    "6_Hard_Shells_Misc":       "./logs_v2/specialists/6_Hard_Shells_Misc/DINOv3Classifier_bystep_2/best_model.pt",
    "7_Detritus_Noise_Empty":   "./logs_v2/specialists/7_Detritus_Noise_Empty/DINOv3Classifier_bystep_2/best_model.pt",
    "8_Triangulaire_Allonge":   "./logs_v2/specialists/8_Triangulaire_Allonge/DINOv3Classifier_bystep_2/best_model.pt"
}

# =========================================================================
# UTILITAIRES INFÉRENCE
# =========================================================================

class InferenceDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error opening {path}: {e}")
            # Retourne une image noire en cas d'erreur pour ne pas crasher
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(path)

def build_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def run_inference(model, images_list, input_size, desc="Inference"):
    dataset = InferenceDataset(images_list, transform=build_transform(input_size))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
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

def get_num_classes_from_checkpoint(ckpt):
    """Extrait le nombre de classes attendu par la dernière couche des poids."""
    keys = list(ckpt.keys())
    head_weights = [k for k in keys if 'head' in k and 'weight' in k]
    if not head_weights:
        return 0
    last_layer_key = head_weights[-1]
    weight_tensor = ckpt[last_layer_key]
    return weight_tensor.shape[0]

# =========================================================================
# MAIN LOOP
# =========================================================================

def main():
    print("🚀 Démarrage de l'inférence hiérarchique (Dataset-Aware)")

    if not os.path.exists(TRAIN_DATA_ROOT):
        print(f"❌ ERREUR : Dossier racine d'entraînement introuvable : {TRAIN_DATA_ROOT}")
        return

    # 1. SCAN IMAGES DE TEST
    all_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "**", "*.*"), recursive=True)
    valid_exts = {'.jpg', '.png', '.jpeg', '.tif', '.bmp'}
    all_images = [x for x in all_images if os.path.splitext(x)[1].lower() in valid_exts]
    print(f"📂 Images à traiter : {len(all_images)}")

    if not all_images:
        return

    # 2. GÉNÉRALISTE
    print("\n========= 1. GENERALIST PREDICTION =========")
    with open(GENERALIST['config'], 'r') as f:
        gen_conf = yaml.safe_load(f)
    gen_input_size = gen_conf['data'].get('images_resize', 224)
    gen_classes = GENERALIST['classes_order']
    
    model_gen = build_model(gen_conf['model'], gen_input_size, len(gen_classes))
    model_gen.load_state_dict(torch.load(GENERALIST['weights'], map_location=DEVICE))
    model_gen.to(DEVICE)
    
    gen_preds_raw = run_inference(model_gen, all_images, gen_input_size, desc="Generalist")
    
    # Nettoyage mémoire
    del model_gen
    torch.cuda.empty_cache()
    
    # Dispatching
    dispatcher = {c: [] for c in gen_classes}
    # Map filename -> fullpath pour récupération facile
    fname_to_path = {os.path.basename(p): p for p in all_images}
    
    for fname, idx in gen_preds_raw.items():
        dispatcher[gen_classes[idx]].append(fname_to_path[fname])

    # 3. SPÉCIALISTES
    print("\n========= 2. SPECIALIST PREDICTION =========")
    
    with open(SPECIALIST_TEMPLATE_YAML, 'r') as f:
        template_conf = yaml.safe_load(f)
    spec_input_size = template_conf['data'].get('images_resize', 224)

    final_results = {}

    for gen_class_name, img_paths in dispatcher.items():
        if not img_paths:
            continue
            
        print(f"\n🔵 Classe Mère : {gen_class_name} ({len(img_paths)} images)")
        
        # Cas 1 : Pas de spécialiste défini dans le dictionnaire
        if gen_class_name not in SPECIALISTS_WEIGHTS:
            print(f"   -> Pas de poids configurés. Label final = {gen_class_name}")
            for p in img_paths: final_results[os.path.basename(p)] = gen_class_name
            continue
        
        weights_path = SPECIALISTS_WEIGHTS[gen_class_name]
        if not os.path.exists(weights_path):
             print(f"   -> ⚠️ Fichier poids introuvable: {weights_path}. Fallback.")
             for p in img_paths: final_results[os.path.basename(p)] = gen_class_name
             continue

        # Cas 2 : Récupération des sous-classes VIA SpecialistDataset
        target_train_dir = os.path.join(TRAIN_DATA_ROOT, gen_class_name)
        
        sub_classes = []
        try:
            # INSTANCIATION SANS TRANSFORM JUSTE POUR LE PARSING DES CLASSES
            # C'est ici que la magie opère : il scanne les fichiers et fait le split('_')
            temp_dataset = SpecialistDataset(root_dir=target_train_dir, transform=None)
            sub_classes = temp_dataset.classes # Liste triée des labels (ex: ['030', '031'])
            print(f"   -> Dataset scanné. {len(sub_classes)} classes trouvées (ex: {sub_classes[:3]}...)")
        except Exception as e:
            print(f"   -> ⚠️ Erreur lors du scan du dossier d'entraînement : {e}")
            print(f"   -> Dossier visé : {target_train_dir}")
            # Fallback
            for p in img_paths: final_results[os.path.basename(p)] = gen_class_name
            continue

        # Validation de cohérence avec les poids
        ckpt = torch.load(weights_path, map_location=DEVICE)
        num_classes_ckpt = get_num_classes_from_checkpoint(ckpt)
        
        if len(sub_classes) != num_classes_ckpt:
            print(f"   ❌ DISMATCH : Le dataset trouve {len(sub_classes)} classes, mais le modèle en attend {num_classes_ckpt}.")
            print("   -> Fallback au généraliste.")
            for p in img_paths: final_results[os.path.basename(p)] = gen_class_name
            del ckpt
            continue

        # Inférence Spécialiste
        model_spec = build_model(template_conf['model'], spec_input_size, len(sub_classes))
        model_spec.load_state_dict(ckpt)
        model_spec.to(DEVICE)
        
        spec_preds = run_inference(model_spec, img_paths, spec_input_size, desc="   -> Specialist")
        
        for fname, idx in spec_preds.items():
            predicted_label_str = sub_classes[idx]
            # Si le label est juste "030", peut-être veux-tu le concaténer avec la classe mère ?
            # Pour l'instant on garde le label brut retourné par le dataset spécialiste.
            final_results[fname] = predicted_label_str
            
        del model_spec
        del ckpt
        torch.cuda.empty_cache()

    # 4. SAUVEGARDE
    print("\n========= 3. SAVING =========")
    df = pd.DataFrame(list(final_results.items()), columns=['imgname', 'label'])
    df = df.sort_values(by='imgname')
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Terminé. Résultats dans : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
