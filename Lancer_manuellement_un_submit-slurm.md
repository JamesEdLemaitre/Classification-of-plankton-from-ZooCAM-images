echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Training"
# Pour train on fait la commande suivante
python -m src.torchtmpl.main config_Resnet50.yaml train
# Ce n'est pas : (car c'est pas le bon dossier, c'est un dossier du même nom
python -m torchtmpl.main config_Resnet50.yaml train
