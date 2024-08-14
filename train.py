import os
import torch
from models.eyenet import EyeNet
from datasets.unity_eyes import UnityEyesDataset
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from tqdm import tqdm
import argparse
import dlib

# Set up pytorch
torch.backends.cudnn.enabled = False
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Check CUDA availability
if device.type == 'cuda':
    print('CUDA is available. Number of GPUs:', torch.cuda.device_count())
    print('Current CUDA device:', torch.cuda.current_device())
    print('GPU name:', torch.cuda.get_device_name(torch.cuda.current_device()))

# Set up cmdline args
parser = argparse.ArgumentParser(description='Trains an EyeNet model using transfer learning')
parser.add_argument('--nstack', type=int, default=3, help='Number of hourglass layers.')
parser.add_argument('--nfeatures', type=int, default=32, help='Number of feature maps to use.')
parser.add_argument('--nlandmarks', type=int, default=34, help='Number of landmarks to be predicted.')
parser.add_argument('--nepochs', type=int, default=10, help='Number of epochs to iterate over all training examples.')
parser.add_argument('--start_from', help='A model checkpoint file to begin training from. This overrides all other arguments.')
parser.add_argument('--pretrained_model', help='A pre-trained model checkpoint file to start transfer learning.')
parser.add_argument('--out', default='checkpoint.pt', help='The output checkpoint filename')
args = parser.parse_args()

class CustomEyeNet(EyeNet):
    def __init__(self, nstack, nfeatures, nlandmarks):
        super(CustomEyeNet, self).__init__(nstack, nfeatures, nlandmarks)
        # Modify the final layer to output 2 features instead of 3
        self.gaze_fc2 = nn.Linear(in_features=256, out_features=2)  # Assuming you want pitch, yaw

# Load pre-trained model
eyenet = CustomEyeNet(nstack=args.nstack, nfeatures=args.nfeatures, nlandmarks=args.nlandmarks).to(device)

# Load facial landmarks predictor
landmarks_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load dataset
dataset = UnityEyesDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load optimizer and loss functions
optimizer = Adam(eyenet.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# Load checkpoint if available
if args.start_from:
    checkpoint = torch.load(args.start_from, map_location=device)
    eyenet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Checkpoint saving function
def save_checkpoint(model, optimizer, epoch, batch, out_file):
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, out_file)
    print(f'Checkpoint saved to {out_file}')

# Training function
def train_epoch(epoch, model, train_loader, optimizer, loss_fn, checkpoint_interval=40):
    model.train()
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch+1}')):
        imgs = batch['img'].to(device)
        heatmaps = batch['heatmaps'].to(device)
        landmarks = batch['landmarks'].to(device)
        gaze = batch['gaze'].to(device)
        
        optimizer.zero_grad()
        heatmaps_pred, landmarks_pred, gaze_pred = model(imgs)
        
        loss = loss_fn(gaze_pred, gaze)
        loss.backward()
        optimizer.step()

        # Save checkpoint every 40 batches
        if (batch_idx + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, batch_idx + 1, args.out)

# Validation function
def validate(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['img'].to(device)
            heatmaps = batch['heatmaps'].to(device)
            landmarks = batch['landmarks'].to(device)
            gaze = batch['gaze'].to(device)
            
            heatmaps_pred, landmarks_pred, gaze_pred = model(imgs)
            val_loss += loss_fn(gaze_pred, gaze).item()
    
    return val_loss / len(val_loader)

# Training loop
best_val_loss = float('inf')
checkpoint_interval = 40

for epoch in range(args.nepochs):
    train_epoch(epoch, eyenet, train_loader, optimizer, loss_fn, checkpoint_interval)
    val_loss = validate(eyenet, val_loader, loss_fn)
    print(f'Validation Loss after epoch {epoch+1}: {val_loss:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(eyenet, optimizer, epoch, 'end', f'{args.out}_best')
        print(f'Best model saved to {args.out}_best')

print('Training complete.')
