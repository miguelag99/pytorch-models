import torch
import urllib
import os
import warnings
import numpy as np
from PIL import Image
from torchvision import transforms

def test_resnet():

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()

    # import urllib
    # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "test/dog.jpg")
    # try: urllib.URLopener().retrieve(url, filename)
    # except: urllib.request.urlretrieve(url, filename)

    # input_image = Image.open(filename)
    input_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        assert output.shape == torch.Size([1, 1000])


