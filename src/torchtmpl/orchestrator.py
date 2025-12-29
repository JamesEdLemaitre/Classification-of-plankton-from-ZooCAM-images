import os
import yaml
import math
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ================= CONFIGURATION =================

MAX_CONCURRENT_JOBS = 2       
MAX_IMAGES_PER_TRAIN = 150000 
PYTHON_EXEC = "python"       

train_num = 2

TRAIN_MODULE = "torchtmpl.main" 
TEMPLATE_PATH = f"config_orchestrator_{train_num}.yaml" 
ORCHESTRATION_LOG_DIR = f"./results/orchestration_{train_num}"

CLASSES_LIST = [
    "1_Copepoda_Torpille",
    "2_Gelatinous_Transparent",
    "3_Macro_Crustacea_Larvae",
    "4_Spherical_Compact_Eggs",
    "5_Elongated_and_Chains",
    "6_Hard_Shells_Misc",
    "7_Detritus_Noise_Empty",
    "8_Triangulaire_Allonge"
]

# ================= HELPER FUNCTIONS =================

def count_images(directory):
    path = Path(directory)
    if not path.exists():
        return 0
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = sum(1 for p in path.rglob('*') if p.suffix.lower() in extensions)
    return count

# ================= FONCTION JOB =================

def run_training_job(class_name):
    print(f"⏳ [Pending] {class_name}")

    train_dir = f"./data/data_reannotated_v2/train/{class_name}"
    safe_log_dir = f"./logs_v2/specialists_{train_num}/{class_name}"
    log_file_path = os.path.join(ORCHESTRATION_LOG_DIR, f"{class_name}.log")

    num_images = count_images(train_dir)
    if num_images == 0:
        print(f"⚠️ [Warning] {class_name} : Aucune image trouvée !")
        divider = 1
    elif num_images > MAX_IMAGES_PER_TRAIN:
        divider = math.ceil(num_images / MAX_IMAGES_PER_TRAIN)
        print(f"ℹ️  [Info] {class_name} : {num_images} images -> Divider : {divider}")
    else:
        divider = 1

    if not os.path.exists(TEMPLATE_PATH):
        print(f"⛔ ERREUR : {TEMPLATE_PATH} introuvable.")
        return

    with open(TEMPLATE_PATH, 'r') as f:
        config = yaml.safe_load(f)

    config['data']['trainpath'] = train_dir
    config['data']['dataset_size_divider'] = divider 
    config['logging']['logdir'] = safe_log_dir
    config['logging']['model_path'] = f"{safe_log_dir}/best_model.pt"

    temp_config_name = f"config_{class_name}.yaml"
    with open(temp_config_name, 'w') as f:
        yaml.dump(config, f)

    cmd = [PYTHON_EXEC, "-m", TRAIN_MODULE, temp_config_name, "train"]

    print(f"🚀 [Start] {class_name} -> Logs propres dans : {log_file_path}")

    # --- MODIFICATION ICI ---
    # On prépare un environnement qui contient une variable spéciale
    custom_env = os.environ.copy()
    custom_env["DISABLE_TQDM"] = "True"
    # ------------------------

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        try:
            # On passe env=custom_env
            subprocess.run(
                cmd, 
                check=True, 
                stdout=log_file, 
                stderr=subprocess.STDOUT, 
                env=custom_env 
            )
            print(f"✅ [Done] {class_name}")

        except subprocess.CalledProcessError as e:
            print(f"❌ [Error] {class_name} (Code {e.returncode}). Voir log : {log_file_path}")
            log_file.write(f"\n\n!!! PROCESS FAILED WITH CODE {e.returncode} !!!\n")

        finally:
            if os.path.exists(temp_config_name):
                os.remove(temp_config_name)

# ================= MAIN =================

if __name__ == "__main__":
    os.makedirs(ORCHESTRATION_LOG_DIR, exist_ok=True)
    print(f"=== Orchestrator : {MAX_CONCURRENT_JOBS} jobs // Logs compacts ===")

    try:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
            list(executor.map(run_training_job, CLASSES_LIST))
    except KeyboardInterrupt:
        print("\n🛑 Interruption demandée...")
    print("=== Terminé ===")
