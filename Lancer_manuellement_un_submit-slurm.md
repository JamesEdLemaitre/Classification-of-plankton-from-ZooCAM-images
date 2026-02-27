echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .

echo "Training"
# Pour train on fait la commande suivante
python -m src.torchtmpl.main config_Resnet50.yaml train
# Ce n'est pas : (car c'est pas le bon dossier, c'est un dossier du même nom
python -m torchtmpl.main NONCESTPASCA config_Resnet50.yaml train



## Pour copier un results csv d'un noeud sur le home depuis un terminal : 

### En chemin complet ~
(venv) sdim_23@sh00:~/Desktop/Deep learning/Projet/kaggle_challenge_sdi$
cp -r ~/Desktop/Deep\ learning/Projet/kaggle_challenge_sdi/results/submission.csv ~/Desktop/Deep\ learning/Projet/kaggle_challenge_sdi/results/submission_from_node.csv

### En chemin relatif
(venv) sdim_23@sh00:~/Desktop/Deep learning/Projet/kaggle_challenge_sdi$ 
cp -r results/submission.csv ~/Desktop/Deep\ learning/Projet/kaggle_challenge_sdi/results/submission_from_nodetest.csv