import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_image_features(image_filenames):
    model = models.vit_l_32(pretrained=True)
    model.fc = nn.Identity()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.train()
    model.to(device)

    image_features = []
    with torch.set_grad_enabled(True):
        for filename in image_filenames:
            image = Image.open(filename).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0).to(device)
            features = model(input_tensor)
            image_features.append(features.cpu().detach().numpy())

    return np.vstack(image_features)