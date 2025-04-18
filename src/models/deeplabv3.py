import torchvision.models.segmentation as seg_models


def get_deeplabv3(model_name='deeplabv3_resnet50', n_classes=10, pretrained=True):
    """
    Loads DeepLabV3 or DeepLabV3+ from torchvision.

    Args:
        model_name (str): 'deeplabv3_resnet50' or 'deeplabv3_mobilenet_v3_large'
        n_classes (int): Number of output classes
        pretrained (bool): Whether to use ImageNet pretrained encoder

    Returns:
        model (nn.Module): DeepLabV3 model with adjusted output layer
    """
    if model_name == 'deeplabv3_resnet50':
        model = seg_models.deeplabv3_resnet50(pretrained=pretrained)
    elif model_name == 'deeplabv3_mobilenet':
        model = seg_models.deeplabv3_mobilenet_v3_large(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Replace the classifier to match number of classes
    model.classifier[4] = seg_models.deeplabv3.DeepLabHead(256, n_classes)
    return model
