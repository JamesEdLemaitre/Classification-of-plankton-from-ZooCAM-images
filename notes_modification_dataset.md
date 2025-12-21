

Je propose une modification du dataset pour faire une pipeline en deux phases.

La phase 1 correspond à faire une classification plus facile. On regroupes les images par caractéristique distinguantes claires:
- 1_Copepoda_Torpille (21 classes)
-> classes qui ressemble à des individus bien développés avec des sortes de tentacules (pas des pattes commes des crustacés en tout cas)
- 2_Gelatinous_Transparent (15 classes)
-> classe qui ressemble à une membrane transparente
- 3_Macro_Crustacea_Larvae (12 classes)
-> classe qui ressemble à des crevettes
- 4_Spherical_Compact_Eggs (15 classes)
-> classe qui rassembles les yeux sphériques, mais aussi j'y ai mis des classes que je ne savais pas où classifier ailleurs mais qui on bien quelque chose de "sphérique" ou en tout cas un peu en cercle
- 5_Elongated_and_Chains (11 classes)
-> classe qui regroupe les éléments allongés
- 6_Hard_Shells_Misc (5 classes)
-> classe qui rassemble des trucs un peu opaques que je savais pas forcément classer ailleurs
- 7_Detritus_Noise_Empty (5 classes)
-> les détritus, le bruit, les cacas
- 8_Triangulaire_Allonge (2 classes)
-> classe qui regroupe les classes type triangulaire/alongé, c'est une classe que j'ai inventé pour éviter de relabeler les images dedans car certaines sont allongé, certaines sont des triangles, certaines un entre deux. Pas simple simple.


J'ai également réannoté à la main la classe 49 qui était soit cercle + tentacules soit des membranes transparentes. La phase 2 réunifiera les deux représentations, l'idée étant que le modèle apprenne "Dans un premier temps je remarque que l'image mo donne une membrane/un cercle tentaculeux; je sais que si c'est comme ça c'est la classe 49" et réunifie donc en phase deux cette classe. 
On la met manuellement dans les données réannotées comme:

cd .../kaggle_challenge_sdi
cp -r ./data/049_01_translucide/* ./data_reannotated/train/2_Gelatinous_Transparent/
cp -r ./data/049_02_pieuvre_like/* ./data_reannotated/train/6_Hard_Shells_Misc/


___ Après analyse des résultats des entraînements des modèles spécialisés:

002 pourrait mériter un split similaire à 49 entre elongated et gelatinous
