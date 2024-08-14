import torch
from datasets.unity_eyes import UnityEyesDataset
from models.eyenet import EyeNet
import os
import numpy as np
import cv2
from util.preprocess import gaussian_2d
from matplotlib import pyplot as plt
from util.gaze import draw_gaze, angular_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomEyeNet(EyeNet):
    def __init__(self, nstack, nfeatures, nlandmarks):
        super(CustomEyeNet, self).__init__(nstack, nfeatures, nlandmarks)
        # Modify the final layer to output 2 features instead of 3
        self.gaze_fc2 = nn.Linear(in_features=256, out_features=2)  # Assuming you want pitch, yaw

def load_model(nstack=3, nfeatures=32, nlandmarks=34):
    checkpoint = torch.load('checkpoint.pt', map_location=device)
    eyenet = CustomEyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
    eyenet.load_state_dict(checkpoint['model_state_dict'])
    return eyenet

def radians_to_degrees(radians):
    return np.degrees(radians)

def process_sample(eyenet, sample):
    with torch.no_grad():
        x = torch.from_numpy(sample['img']).float().unsqueeze(0).to(device)
        heatmaps_pred, landmarks_pred, gaze_pred = eyenet.forward(x)

        landmarks_pred = landmarks_pred.cpu().numpy()[0, :]
        gaze_pred = gaze_pred.cpu().numpy()[0, :]

        actual_pitch, actual_yaw = radians_to_degrees(sample['gaze'])
        pred_pitch, pred_yaw = radians_to_degrees(gaze_pred)
        error = angular_error(sample['gaze'].reshape(1, -1), gaze_pred.reshape(1, -1))[0]

        return actual_pitch, actual_yaw, pred_pitch, pred_yaw, error, landmarks_pred, gaze_pred

def visualize_sample(sample, landmarks_pred, gaze_pred, actual_pitch, actual_yaw, pred_pitch, pred_yaw, error):
    plt.figure(figsize=(12, 15))

    iris_center = sample['landmarks'][-2][::-1]
    iris_center *= 2
    img = cv2.cvtColor(sample['img'], cv2.COLOR_GRAY2RGB)

    img_gaze_pred = img.copy()
    for (y, x) in landmarks_pred[-2:-1]:
        cv2.circle(img_gaze_pred, (int(x*2), int(y*2)), 2, (255, 0, 0), -1)
    draw_gaze(img_gaze_pred, iris_center, gaze_pred, length=60, color=(255, 0, 0))

    img_gaze = img.copy()
    for (x, y) in sample['landmarks'][-2:-1]:
        cv2.circle(img_gaze, (int(x*2), int(y*2)), 2, (0, 255, 0), -1)
    draw_gaze(img_gaze, iris_center, sample['gaze'], length=60, color=(0, 255, 0))

    plt.subplot(321)
    plt.imshow(cv2.cvtColor(sample['full_img'], cv2.COLOR_BGR2RGB))
    plt.title('Raw training image')

    plt.subplot(322)
    plt.imshow(img, cmap='gray')
    plt.title('Preprocessed training image')

    plt.subplot(323)
    plt.imshow(np.mean(sample['heatmaps'][16:32], axis=0), cmap='gray')
    plt.title('Ground truth heatmaps')

    plt.subplot(324)
    plt.imshow(np.mean([gaussian_2d(w=80, h=48, cx=c[1], cy=c[0], sigma=3) for c in landmarks_pred][16:32], axis=0), cmap='gray')
    plt.title('Predicted heatmaps')

    plt.subplot(325)
    plt.imshow(img_gaze)
    plt.title('Ground truth landmarks and gaze vector')

    plt.subplot(326)
    plt.imshow(img_gaze_pred)
    plt.title('Predicted landmarks and gaze vector')

    # Add text with pitch, yaw, and angular error
    info_text = (
        f"Actual Pitch: {actual_pitch:.2f}°, Yaw: {actual_yaw:.2f}°\n"
        f"Predicted Pitch: {pred_pitch:.2f}°, Yaw: {pred_yaw:.2f}°\n"
        f"Angular Error: {error:.2f}°"
    )
    plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()


def evaluate_dataset(eyenet, dataset):
    actual_pitches, actual_yaws = [], []
    pred_pitches, pred_yaws = [], []
    errors = []

    num_samples = len(dataset)
    num_samples_to_process = int(0.005 * num_samples)

    for i in range(num_samples_to_process):
        sample = dataset[i]
        actual_pitch, actual_yaw, pred_pitch, pred_yaw, error, _, _ = process_sample(eyenet, sample)
        
        actual_pitches.append(actual_pitch)
        actual_yaws.append(actual_yaw)
        pred_pitches.append(pred_pitch)
        pred_yaws.append(pred_yaw)
        errors.append(error)

        if i % 100 == 0:
            print(f"Processed {i}/{num_samples_to_process} samples")

    print(f"Mean Angular Error: {np.mean(errors):.2f}° ± {np.std(errors):.2f}°")
    
    # Calculate accuracy (within 10 degrees)
    threshold = 10
    pitch_accuracy = np.mean(np.abs(np.array(pred_pitches) - np.array(actual_pitches)) < threshold)
    yaw_accuracy = np.mean(np.abs(np.array(pred_yaws) - np.array(actual_yaws)) < threshold)

    # Create confusion matrices
    bins = np.linspace(-90, 90, 19)
    pitch_cm = confusion_matrix(np.digitize(actual_pitches, bins), np.digitize(pred_pitches, bins))
    yaw_cm = confusion_matrix(np.digitize(actual_yaws, bins), np.digitize(pred_yaws, bins))

    plt.figure(figsize=(20, 8))

    plt.subplot(121)
    sns.heatmap(pitch_cm, annot=False, cmap='YlOrRd')
    plt.title(f'Pitch Confusion Matrix\nAccuracy: {pitch_accuracy:.2%}')
    plt.xlabel('Predicted Pitch')
    plt.ylabel('True Pitch')

    plt.subplot(122)
    sns.heatmap(yaw_cm, annot=False, cmap='YlOrRd')
    plt.title(f'Yaw Confusion Matrix\nAccuracy: {yaw_accuracy:.2%}')
    plt.xlabel('Predicted Yaw')
    plt.ylabel('True Yaw')

    plt.tight_layout()
    plt.show()

    print(f"Pitch Accuracy (within {threshold}°): {pitch_accuracy:.2%}")
    print(f"Yaw Accuracy (within {threshold}°): {yaw_accuracy:.2%}")

def process_custom_image(eyenet, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (160, 96))
    img = img / 255.0
    img = torch.from_numpy(img).float().unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, gaze_pred = eyenet.forward(img)
        gaze_pred = gaze_pred.cpu().numpy()[0, :]
        pred_pitch, pred_yaw = radians_to_degrees(gaze_pred)

    print(f"Predicted Pitch: {pred_pitch:.2f}°, Predicted Yaw: {pred_yaw:.2f}°")
    print(f"Predicted Gaze Vector: [{gaze_pred[0]:.4f}, {gaze_pred[1]:.4f}]")

if __name__ == "__main__":
    # Specify the model architecture parameters here
    nstack = 3
    nfeatures = 32
    nlandmarks = 34
    
    eyenet = load_model(nstack, nfeatures, nlandmarks)
    dataset = UnityEyesDataset()

    # Process and visualize a sample
    sample = dataset[2]
    actual_pitch, actual_yaw, pred_pitch, pred_yaw, error, landmarks_pred, gaze_pred = process_sample(eyenet, sample)

    print(f"Sample 2 Results:")
    print(f"Actual Pitch: {actual_pitch:.2f}°, Actual Yaw: {actual_yaw:.2f}°")
    print(f"Predicted Pitch: {pred_pitch:.2f}°, Predicted Yaw: {pred_yaw:.2f}°")
    print(f"Angular Error: {error:.2f}°")
    print(f"Actual Gaze Vector: [{sample['gaze'][0]:.4f}, {sample['gaze'][1]:.4f}]")
    print(f"Predicted Gaze Vector: [{gaze_pred[0]:.4f}, {gaze_pred[1]:.4f}]")

    visualize_sample(sample, landmarks_pred, gaze_pred, actual_pitch, actual_yaw, pred_pitch, pred_yaw, error)

  

    # Evaluate the entire dataset
    print("\nEvaluating entire dataset...")
    evaluate_dataset(eyenet, dataset)

    # Process a custom image
    custom_image_path = 'C:\\Users\\prajw\\Desktop\\Desktop\\Docs\\SummerInternship\\Codes\\gaze-estimation\\datasets\\MPIIGaze\\0134.jpg'
    print(f"\nProcessing custom image: {custom_image_path}")
    process_custom_image(eyenet, custom_image_path)
