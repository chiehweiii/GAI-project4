# Combined DDPM and DIP Model for Image Denoising

This project demonstrates the effectiveness of combining Denoising Diffusion Probabilistic Models (DDPM) and Deep Image Prior (DIP) techniques for image denoising. The approach leverages the strengths of both models to achieve improved image quality and generation speed.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- Google Colab (optional, for running on Google Drive)

## Dataset

This cars dataset contains great training and testing sets for forming models that can tell cars from one another. Data originated from Stanford University AI Lab.
The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. 

1. Prepare your dataset and organize it in the following structure:
    ```
    /path/to/dataset/
        ├── class1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        └── class2/
            ├── img1.jpg
            ├── img2.jpg
            └── ...
    ```

2. For this project, the dataset should be placed in your Google Drive:
    ```
    /content/drive/My Drive/Colab_Notebooks/P4_cars_trains/
    ```

## Acknowledgements

Data source and banner image: http://ai.stanford.edu/~jkrause/cars/car_dataset.html contains all bounding boxes and labels for both training and tests.

3D Object Representations for Fine-Grained Categorization

Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei

4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.

## How to Run

### Step1: Mount Google Drive (For Google Colab)

If using Google Colab, mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```
### Step2: Define and Train DIP Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
class DIPModel(nn.Module):
    def __init__(self):
        super(DIPModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.layers(x)

# Training parameters
img_size = 64
num_epochs_dip = 1000
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Get one sample image
dataiter = iter(dataloader)
clean_image, _ = next(dataiter) 
clean_image = clean_image.to(device)

# Add noise
noisy_image = add_noise(clean_image).to(device)

# Initialize DIP model
dip_model = DIPModel().to(device)
optimizer_dip = optim.Adam(dip_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Train DIP
trained_dip_model = train_dip(dip_model, noisy_image, clean_image, num_epochs_dip, optimizer_dip, criterion)

```

### Step3: Define and Train DDPM Model

```python
class DDPM(nn.Module):
    def __init__(self, timesteps=1000):
        super(DDPM, self).__init__()
        self.timesteps = timesteps
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, t):
        return self.model(x) * (1 - t / self.timesteps)

# DIP output = initial prior for DDPM
initial_prior = trained_dip_model(noisy_image).detach()

# Initialize DDPM model
ddpm_model = DDPM().to(device)
optimizer_ddpm = optim.Adam(ddpm_model.parameters(), lr=learning_rate)

# Train DDPM
trained_ddpm_model = train_ddpm(ddpm_model, initial_prior, num_epochs_ddpm, optimizer_ddpm, criterion)


```

### Step4: Display Results
```python
show_image(noisy_image, title="Noisy Image")
show_image(initial_prior, title="DIP Initial Prior")
show_image(trained_ddpm_model(initial_prior, trained_ddpm_model.timesteps), title="DDPM Final Output")
show_image(clean_image, title="Original Image")

```


