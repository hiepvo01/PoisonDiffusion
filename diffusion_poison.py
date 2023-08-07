import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_loader = torch.utils.data.DataLoader(cifar_trainset, batch_size=64, shuffle=True)

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Use a pretrained ResNet for feature extraction
        self.resnet = models.resnet50(pretrained=True)
        # Remove the last linear layer to get features
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
    
    def forward(self, x):
        return self.features(x)
    
# class FeatureExtractor(nn.Module):
    # def __init__(self, model_name="resnet18"):
    #     super(FeatureExtractor, self).__init__()
    #     if model_name == "resnet18":
    #         self.model = models.resnet18(pretrained=True)
    #     else:
    #         raise ValueError("Unsupported model!")
    #     self.features = nn.Sequential(*list(self.model.children())[:-1])
    
    # def forward(self, x):
    #     return self.features(x).squeeze()

    
    
def adversarial_loss(predicted, original, target_features, feature_extractor, alpha=0.5):
    reconstruction_loss = ((predicted - original) ** 2).mean()
    predicted_features = feature_extractor(predicted)
    feature_loss = ((predicted_features - target_features) ** 2).mean()
    return alpha * reconstruction_loss + (1 - alpha) * feature_loss


class DenoisingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DenoisingModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, t):
        # We concatenate the time variable to let the model know at which timestep the denoising is occurring
        xt = torch.cat([x, t * torch.ones_like(x)], dim=1)
        return self.net(xt)

def get_beta(t, T):
    # This is a very basic beta schedule
    # The original paper and subsequent implementations have a more nuanced beta schedule
    return (1 - (t / T)) * 0.01

def train_adversarial_ddpm(model, feature_extractor, data_loader, target_image, optimizer, num_steps, device='cuda'):
    model.train()
    model.to(device)
    feature_extractor.to(device)
    target_features = feature_extractor(target_image).detach()

    for epoch in range(num_steps):
        for x in data_loader:
            x = x.to(device)
            
            # Sample a random timestep
            t = torch.randint(0, num_steps, (x.size(0), 1), device=device).float() / num_steps
            
            # Noising process
            noise = torch.randn_like(x) * get_beta(epoch, num_steps)
            noisy_x = x + noise
            
            # Predict denoised version
            predicted = model(noisy_x, t)
            
            # Compute the adversarial loss
            loss = adversarial_loss(predicted, x, target_features, feature_extractor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# Hyperparameters
INPUT_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 1000

# For CIFAR
feature_extractor_cifar = FeatureExtractor().to('cuda')

# Target image for CIFAR
target_image_cifar = next(iter(cifar_loader))[0][0:1].to('cuda')  # taking the first image as target

# DDPM model
model_cifar = DDPM(...).to('cuda')  # define DDPM parameters suitable for CIFAR

optimizer_cifar = torch.optim.Adam(model_cifar.parameters(), lr=0.001)

train_adversarial_ddpm(model_cifar, feature_extractor_cifar, cifar_loader, target_image_cifar, optimizer_cifar, 100)
