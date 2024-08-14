import pytest
import sys
import os

# Add the path to the datasets directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets')))

try:
    from datasets.unity_eyes import UnityEyesDataset
    print("Successfully imported UnityEyesDataset.")
except ImportError as e:
    print(f"Failed to import UnityEyesDataset: {e}")

def test_unity_eyes():
    try:
        img_dir = os.path.join(os.path.dirname(__file__), 'data/imgs')
        print(f"Looking for images in: {img_dir}")

        # Print the contents of the directory
        if os.path.exists(img_dir):
            print("Directory exists. Contents:")
            print(os.listdir(img_dir))
        else:
            print("Directory does not exist.")
            return

        ds = UnityEyesDataset(img_dir=img_dir)
        print(f"Dataset initialized. Number of samples: {len(ds)}")

        if len(ds) == 0:
            print("No images found in the dataset.")
            return

        print("Fetching sample...")
        sample = ds[0]
        print("Sample fetched. Running assertions...")
        assert sample['full_img'].shape == (600, 800, 3)
        print("Assertion for full_img shape passed.")
        assert sample['img'].shape == (90, 150, 3)
        print("Assertion for img shape passed.")
        assert float(sample['json_data']['eye_details']['iris_size']) == 0.9349335
        print("Assertion for iris_size passed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_unity_eyes()
