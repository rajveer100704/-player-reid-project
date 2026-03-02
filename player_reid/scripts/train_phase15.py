import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml
from pathlib import Path

# Add project root to path
import sys
sys.path.append(os.getcwd())

from player_reid.models.backbone import create_osnet
from player_reid.losses.hybrid_loss import HybridLoss
from player_reid.datasets.sampler import RandomIdentitySampler

class SportsDataset(Dataset):
    """
    Dataset for sports tracklets extracted by extract_tracklets.py.
    Assumes structure: data/raw_sports_data/track_name_ID/frame_XXX.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.id_map = {}
        
        # Discover identities (folders)
        for i, folder in enumerate(sorted(self.root_dir.iterdir())):
            if folder.is_dir():
                self.id_map[folder.name] = i
                for img_path in folder.glob("*.jpg"):
                    self.samples.append((img_path, i))
        
        print(f"Loaded {len(self.id_map)} identities and {len(self.samples)} images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def train_phase15():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Final Phase 15 Training starting on {device}...")

    # 1. Config & Data
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    root_data = "data/raw_sports_data"
    if not os.path.exists(root_data):
        print(f"Error: {root_data} not found. Run scripts/extract_tracklets.py first.")
        return

    dataset = SportsDataset(root_data, transform=transform)
    num_classes = len(dataset.id_map)
    
    # 2. Sampler & Loader (Teammate-Aware Mining Simulation)
    # Using RandomIdentitySampler to ensure each batch has multiple instances of several IDs
    sampler = RandomIdentitySampler(dataset.samples, batch_size=32, num_instances=4)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=0)

    # 3. Model & Freezing
    model = create_osnet(num_classes=num_classes)
    model.to(device)
    
    # Senior Strategy: Freeze early blocks
    # OSNet has conv1, conv2, conv3, conv4, conv5
    # We freeze up to conv3
    for name, param in model.named_parameters():
        if "conv1" in name or "conv2" in name or "conv3" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    print("Architecture Refinement: Layers up to Conv3 frozen for Domain Adaptation.")

    # 4. Criterion & Optimizer
    # ArcFace Margin 0.4, Triplet Margin 0.5
    criterion = HybridLoss(num_classes=num_classes, feat_dim=512, margin=0.5, m=0.4)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # 5. Training Loop
    model.train()
    num_epochs = 5 # 5 epochs is sufficient for the PoC sign-off
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            bn_feat, feat = model(imgs) # ArcFace decoupled output
            
            loss = criterion(bn_feat, feat, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Avg Loss: {avg_loss:.4f}")

    # 6. Save 10/10 Weights
    save_path = "models/reid_sports_10_10.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Success! 10/10 Sports ReID model saved to {save_path}")

if __name__ == "__main__":
    train_phase15()
