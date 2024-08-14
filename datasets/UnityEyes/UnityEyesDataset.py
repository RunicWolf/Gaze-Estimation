import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class UnityEyesDataset(Dataset):
    def __init__(self, img_dir='datasets/UnityEyes/images'):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        print(f"Found {len(self.img_paths)} images.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if idx >= len(self.img_paths):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.img_paths)} images.")
        
        full_img = cv2.imread(self.img_paths[idx])
        img = cv2.resize(full_img, (160, 96))  # Example preprocessing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = img / 255.0  # Normalize
        img = img.astype(np.float32)

        # Placeholder for other data such as heatmaps, landmarks, and gaze
        heatmaps = np.zeros((34, 48, 80), dtype=np.float32)
        landmarks = np.zeros((34, 2), dtype=np.float32)
        gaze = np.zeros((2,), dtype=np.float32)

        return {
            'full_img': full_img,
            'img': img,
            'heatmaps': heatmaps,
            'landmarks': landmarks,
            'gaze': gaze
        }

# Example usage
if __name__ == '__main__':
    dataset = UnityEyesDataset(img_dir='datasets/UnityEyes/images')
    print(f"Dataset contains {len(dataset)} samples.")
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Sample {i}: {sample['full_img'].shape}")
