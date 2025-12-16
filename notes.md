mettre en place lr-scheduler

tester nn.CrossEntropyLoss(weight=class_weights) pour classes déséquilibrée ?

tester différents paramétrages pour 
sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights)/20, ---> actuellement, on répète certaines images des petites classes et on en dégage beaucoup d'images des grandes classes . 
        replacement=True
    )

