# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn

#appeler un resnet 50 pretrained
def ResNet50(cfg, input_size, num_classes):
    """
    ResNet50 model from torchvision.models
    """
    from torchvision.models import resnet50, ResNet50_Weights

    # Load a pretrained ResNet50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Modify the first convolutional layer if the input has a different number of channels
    if input_size[0] != 3:
        model.conv1 = nn.Conv2d(
            input_size[0],
            64, #nombre de canaux de sorties
            kernel_size=7,#filtre de dimension 7*7
            stride=2,
            padding=3,
            bias=False,
        )

    print("nombre de channel : ", input_size[0])

    # Modify the final fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

#appeler un resnet 50 pretrained
def ResNet34(cfg, input_size, num_classes):
    """
    ResNet34 model from torchvision.models
    """
    from torchvision.models import resnet34, ResNet34_Weights

    # Load a pretrained ResNet50 model
    model = ResNet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    # Modify the first convolutional layer if the input has a different number of channels
    if input_size[0] != 3:
        model.conv1 = nn.Conv2d(
            input_size[0],
            64, #nombre de canaux de sorties
            kernel_size=7,#filtre de dimension 7*7
            stride=2,
            padding=3,
            bias=False,
        )

    print("nombre de channel : ", input_size[0])

    # Modify the final fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def efficientnet_b7(cfg, input_size, num_classes):
    import timm

    from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

    weights = EfficientNet_B7_Weights.IMAGENET1K_V1
    model = efficientnet_b7(weights=weights)
    #model = timm.create_model('efficientnet_b7', pretrained=True)

    #print(model.features[0][0])  # Première couche
    #print(model.classifier[1])   # Dernière couche linéaire

    model.features[0][0] = nn.Conv2d(
        input_size[0],
        model.features[0][0].out_channels, #32 en dur
        kernel_size=model.features[0][0].kernel_size[0],
        stride=model.features[0][0].stride[0],
        padding=model.features[0][0].padding[0],
        bias=False, ## si on fait de la batch normalization le bias sert a rien
    )

    print("nombre de channel : ", input_size[0])

    # Modify the final fully connected layer to match the number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model



def efficientnet_b5(cfg, input_size, num_classes):
    import timm

    from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights

    weights = EfficientNet_B5_Weights.IMAGENET1K_V1
    model = efficientnet_b5(weights=weights)

    # ✅ Sauvegarder les poids pré-entraînés AVANT de remplacer
    pretrained_weight = model.features[0][0].weight.data  # shape: [32, 3, 3, 3]

    print(model.features[0][0])  # Première couche
    #print(model.classifier[1])   # Dernière couche linéaire
    #print(model.classifier)   # Dernière couche linéaire

    if input_size[0] == 1:
        print("Input size égal à 1, les images sont en nuances de gris")
        model.features[0][0] = nn.Conv2d(
            input_size[0],
            model.features[0][0].out_channels, #32 en dur
            kernel_size=model.features[0][0].kernel_size[0],
            stride=model.features[0][0].stride[0],
            padding=model.features[0][0].padding[0],
            bias=False, ## si on fait de la batch normalization le bias sert a rien
        )

        # ✅ Initialiser avec la moyenne des 3 canaux RGB pré-entraînés
        with torch.no_grad():
            model.features[0][0].weight = nn.Parameter(
                pretrained_weight.mean(dim=1, keepdim=True)  # shape: [32, 1, 3, 3]
            )
    elif input_size[0] == 3:
        print("Input size égal à 3, les images sont en RGB. \nNe pas toucher à la première couche de convolution si on part du modèle préentrainé")
    else : 
        print("Le input size n'est ni 1 ni 3, implémenter une nouvelle façon de transferer les poids préentrainé en nouvelle dimension")

    print("nombre de channel : ", input_size[0])

    # Modify the final fully connected layer to match the number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model