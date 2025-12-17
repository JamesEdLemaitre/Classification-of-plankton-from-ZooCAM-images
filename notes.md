mettre en place lr-scheduler

tester nn.CrossEntropyLoss(weight=class_weights) pour classes déséquilibrée ?

tester différents paramétrages pour 
sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights)/20, ---> actuellement, on répète certaines images des petites classes et on en dégage beaucoup d'images des grandes classes . 
        replacement=True
    )

Questionnements:
- Impact de différents denoising (nn ou filtre gaussien ou techniques de traitement d'image traditionnelles plus avancées) sur les performances, en particulier comparer les matrices de confusion: est-ce que toutes les classes bénéficient de ce preprocessing ? Est-ce que toutes les dimensions d'images en bénéficient ? 
- Impact du resizing "naïf" des images: est-ce que les images ayant des dimensions et des orientations variables, un resizing simple sans padding ni rien est optimal ou proche de l'optimal ? 
- Est-ce qu'entraîner deux têtes de classification en deux étapes améliorerait les perfs:
-- Première étape: entraîner des catégories plus large (oeuf, intermédiaire, adulte, ...) puis freeze ces paramètres et passer à
-- Deuxième étape: entraîner à prédire les sous catégories sur une deuxième tête de classification

- Post-processing possible ? Pas sûr de la pertinence pour ce challenge mais à regarder

