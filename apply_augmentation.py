"""
import matplotlib.pyplot as plt
import os
from torchio import Image, Subject, RandomNoise, RandomMotion, RandomBiasField, RandomGamma, Compose
from PIL import Image as PILImage
import numpy as np
import torchio as tio
from torchvision import transforms
import math

def augment_image(image_path, augmentations):
    original_image = PILImage.open(image_path).convert('L')  # Convert to grayscale
    original_tensor = transforms.ToTensor()(original_image)
    original_tensor = original_tensor[0, :, :]

    # Check if RandomMotion is in the augmentations
    motion_aug = next((aug for aug in augmentations if isinstance(aug, RandomMotion)), None)
    
    if motion_aug:
        # For motion artifacts
        adjusted_tensor = original_tensor.unsqueeze(0).unsqueeze(-1)
        subject = Subject(image=tio.ScalarImage(tensor=adjusted_tensor))
        
        # Apply RandomMotion
        augmented_subject = motion_aug(subject)
        random_motion = augmented_subject.history[0]
        
        # Apply the random motion to a differently shaped tensor
        adjusted_tensor_ = original_tensor.unsqueeze(0).unsqueeze(0)
        subject_ = Subject(image=tio.ScalarImage(tensor=adjusted_tensor_))
        augmented_subject = random_motion(subject_)
        
        # Apply other augmentations if any
        other_augs = [aug for aug in augmentations if not isinstance(aug, RandomMotion)]
        if other_augs:
            transform = Compose(other_augs)
            augmented_subject = transform(augmented_subject)
    else:
        # For non-motion augmentations
        adjusted_tensor = original_tensor.unsqueeze(0).unsqueeze(0)
        subject = Subject(image=tio.ScalarImage(tensor=adjusted_tensor))
        
        # Apply all augmentations
        transform = Compose(augmentations)
        augmented_subject = transform(subject)
    
    augmented_image_numpy = augmented_subject.image.data.numpy()
    image_data = augmented_image_numpy[0, 0, :, :]
    
    return image_data  # Return the augmented image data as numpy array

def save_augmented_image(augmented_image, save_path):
    plt.imshow(augmented_image, cmap='gray' if augmented_image.ndim == 2 else None)
    plt.axis('off')  # Hide axis
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free up memory

def stitch_images(image_paths, output_path, max_cols=8):
    images = [PILImage.open(img_path) for img_path in image_paths]
    num_images = len(images)
    cols = min(num_images, max_cols)
    rows = math.ceil(num_images / cols)

    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)

    stitched_image = PILImage.new('RGB', (max_width * cols, max_height * rows))

    for i, img in enumerate(images):
        x = (i % cols) * max_width
        y = (i // cols) * max_height
        stitched_image.paste(img, (x, y))

    stitched_image.save(output_path)

def process_dataset(dataset_path, output_path):
    augmentation_strengths = {
        'weak': {
            'Noise': RandomNoise(mean=0, std=0.01),
            'Motion': RandomMotion(degrees=(-10, 10), translation=(-10, 10), num_transforms=2),
            'BiasField': RandomBiasField(coefficients=0.1),
            'Gamma': RandomGamma(log_gamma=(-0.2, 0.1))
        },
        'medium': {
            'Noise': RandomNoise(mean=0, std=0.06),
            'Motion': RandomMotion(degrees=(-90, 90), translation=(-90, 90), num_transforms=8),
            'BiasField': RandomBiasField(coefficients=0.3),
            'Gamma': RandomGamma(log_gamma=(0.1, 0.3))
        },
        'strong': {
            'Noise': RandomNoise(mean=0, std=0.15),
            'Motion': RandomMotion(degrees=(-140, 140), translation=(-140, 140), num_transforms=16),
            'BiasField': RandomBiasField(coefficients=0.6),
            'Gamma': RandomGamma(log_gamma=(0.3, 0.8))
        },
    }

    valid_image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    augmented_images = {level: {aug_type: [] for aug_type in list(augmentation_strengths['weak'].keys()) + ['All']} 
                        for level in ['weak', 'medium', 'strong']}
    
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                
                if not os.path.splitext(image_name)[1].lower() in valid_image_extensions:
                    continue
                
                # Apply each individual augmentation and save the results
                for level in ['weak', 'medium', 'strong']:
                    # Individual augmentations
                    for aug_type, transform in augmentation_strengths[level].items():
                        augmented_image = augment_image(image_path, [transform])
                        
                        output_dir = os.path.join(output_path, aug_type, level, folder_name)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        save_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_{aug_type}_{level}.png")
                        save_augmented_image(augmented_image, save_path)
                        augmented_images[level][aug_type].append(save_path)
                    
                    # Combined augmentation
                    combined_transforms = list(augmentation_strengths[level].values())
                    combined_augmented_image = augment_image(image_path, combined_transforms)
                    
                    output_dir = os.path.join(output_path, 'All_Augmentations', level, folder_name)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    save_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_All_{level}.png")
                    save_augmented_image(combined_augmented_image, save_path)
                    augmented_images[level]['All'].append(save_path)

    # Stitch together augmented images for each augmentation type and strength
    for level in ['weak', 'medium', 'strong']:
        for aug_type in augmented_images[level].keys():
            stitched_dir = os.path.join(output_path, 'Stitched', aug_type, level)
            os.makedirs(stitched_dir, exist_ok=True)
            stitched_path = os.path.join(stitched_dir, f"{aug_type}_{level}_stitched.png")
            
            stitch_images(augmented_images[level][aug_type], stitched_path)

# Example usage
dataset_path = '/Users/a86153/Desktop/VLM_evaluation/pneumonia'  # Replace with your dataset path
output_path = '/Users/a86153/Desktop/VLM_evaluation/pneumonia augmented'  # Replace with your output path

process_dataset(dataset_path, output_path)
"""

import matplotlib.pyplot as plt
import os
import torch
import torchio as tio
from torchio import Image, Subject, RandomNoise, RandomMotion, RandomBiasField, RandomGamma, Compose, RandomAffine
from PIL import Image as PILImage
import numpy as np
from torchvision import transforms
import math
from scipy.ndimage import rotate

# Create a TorchIO transform that uses the custom_crop function
import numpy as np
import torch
import torchio as tio

class RandomCrop(tio.Transform):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def apply_transform(self, subject):
        image = subject.image
        data = image.data.squeeze().numpy()  # Remove single dimensions and convert to numpy

        h, w = data.shape
        crop_h = int(h * self.scale_factor)
        crop_w = int(w * self.scale_factor)

        # Ensure crop size doesn't exceed image dimensions
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        # Calculate random start positions within the image bounds
        start_h = np.random.randint(0, h - crop_h + 1)
        start_w = np.random.randint(0, w - crop_w + 1)

        # Perform the crop
        cropped_data = data[start_h:start_h+crop_h, start_w:start_w+crop_w]
        #cropped_data = (cropped_data - np.min(cropped_data)) / (np.max(cropped_data) - np.min(cropped_data))
        
        # Convert back to 4D tensor for TorchIO
        cropped_tensor = torch.from_numpy(cropped_data).float().unsqueeze(0).unsqueeze(0)

        # Create a new image with the cropped data
        cropped_subject = tio.Subject(image=tio.ScalarImage(tensor=cropped_tensor, affine=image.affine))

        return cropped_subject

class FixedRotation(tio.Transform):
    def __init__(self, angle):
        super().__init__()
        self.angle = angle

    def apply_transform(self, subject):
        image = subject.image
        data = image.data.numpy()

        # Squeeze the data to remove single-dimensional entries
        data = np.squeeze(data)

        # Ensure the image is 2D
        if len(data.shape) > 2:
            raise ValueError(f"Unexpected data shape: {data.shape}. Expected 2D image.")

        # Rotate the image
        rotated_data = rotate(data, self.angle, reshape=True, mode='constant', cval=0)

        # Normalize the rotated image to 0-1 range
        #rotated_data = (rotated_data - np.min(rotated_data)) / (np.max(rotated_data) - np.min(rotated_data))

        # Reshape to match TorchIO's expected format (1, 1, H, W)
        rotated_data = rotated_data.reshape(1, 1, *rotated_data.shape)

        # Convert back to torch tensor
        rotated_tensor = torch.from_numpy(rotated_data).float()

        # Create new Subject object with rotated image
        rotated_subject = tio.Subject(image=tio.ScalarImage(tensor=rotated_tensor, affine=image.affine))

        return rotated_subject



def augment_image(image_path, augmentations):
    # Load the image as a grayscale PIL Image
    original_image = PILImage.open(image_path).convert('L')

    # Determine if Crop or Rotate is in the augmentations
    uses_crop_or_rotate = any(isinstance(aug, (RandomCrop, FixedRotation)) for aug in augmentations)

    # Check if RandomMotion is in the augmentations
    motion_aug = next((aug for aug in augmentations if isinstance(aug, RandomMotion)), None)
    
    if uses_crop_or_rotate:
        # For Crop or Rotate, do not use transforms.ToTensor()
        original_array = np.array(original_image, dtype=np.float32)  # Keep original intensities
        original_tensor = torch.from_numpy(original_array)  # Shape: (H, W)
    else:
        # For other augmentations, use transforms.ToTensor()
        original_tensor = transforms.ToTensor()(original_image)  # Shape: (1, H, W), scaled to [0, 1]
        original_tensor = original_tensor[0]  # Remove channel dimension to get shape: (H, W)

    if motion_aug:
        # For motion artifacts
        # Adjust tensor shape for RandomMotion
        adjusted_tensor = original_tensor.unsqueeze(0).unsqueeze(-1)  # Shape: (1, H, W, 1)
        subject = Subject(image=tio.ScalarImage(tensor=adjusted_tensor))
        
        # Apply RandomMotion
        augmented_subject = motion_aug(subject)
        random_motion = augmented_subject.history[0]
        
        # Apply the random motion to a differently shaped tensor
        adjusted_tensor_ = original_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        subject_ = Subject(image=tio.ScalarImage(tensor=adjusted_tensor_))
        augmented_subject = random_motion(subject_)
        
        # Apply other augmentations if any
        other_augs = [aug for aug in augmentations if not isinstance(aug, RandomMotion)]
        if other_augs:
            for aug in other_augs:
                augmented_subject = aug(augmented_subject)
    else:
        # For non-motion augmentations
        adjusted_tensor = original_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        subject = Subject(image=tio.ScalarImage(tensor=adjusted_tensor))
        
        # Apply all augmentations
        for aug in augmentations:
            subject = aug(subject)
        augmented_subject = subject

    # Convert back to numpy array for saving
    augmented_image_numpy = augmented_subject.image.data.numpy()
    image_data = augmented_image_numpy[0, 0, :, :]  # Get the image data

    return image_data  # Return the augmented image data as numpy array

def save_augmented_image(augmented_image, save_path, aug_type):
    if aug_type == 'Crop':
        # For Crop augmentation, use PIL to save the image
        augmented_image = augmented_image.astype(np.uint8)
        cropped_pil_image = PILImage.fromarray(augmented_image, mode='L')
        cropped_pil_image.save(save_path)
    else:
        plt.imshow(augmented_image, cmap='gray' if augmented_image.ndim == 2 else None)
        plt.axis('off')  # Hide axis
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free up memory
    
def stitch_images(image_paths, output_path, max_cols=8):
    images = [PILImage.open(img_path) for img_path in image_paths]
    num_images = len(images)
    cols = min(num_images, max_cols)
    rows = math.ceil(num_images / cols)

    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)

    stitched_image = PILImage.new('RGB', (max_width * cols, max_height * rows))

    for i, img in enumerate(images):
        x = (i % cols) * max_width
        y = (i // cols) * max_height
        stitched_image.paste(img, (x, y))

    stitched_image.save(output_path)

def process_dataset(dataset_path, output_path):
    augmentation_strengths = {
    'weak': {
        'Noise': tio.RandomNoise(mean=0, std=0.10),
        'Motion': tio.RandomMotion(degrees=30, translation=30, num_transforms=8),
        'BiasField': tio.RandomBiasField(coefficients=0.7),
        'Crop': RandomCrop(0.8),  # 80% of original size
        'Rotate': FixedRotation(angle=30)
    },
    'strong': {
        'Noise': tio.RandomNoise(mean=0, std=0.35),
        'Motion': tio.RandomMotion(degrees=160, translation=120, num_transforms=32),
        'BiasField': tio.RandomBiasField(coefficients=1.8),
        'Crop': RandomCrop(0.4),  # 20% of original size
        'Rotate': FixedRotation(angle=90)
    },
}

    valid_image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    augmented_images = {level: {aug_type: [] for aug_type in list(augmentation_strengths['weak'].keys())} 
                        for level in ['weak', 'strong']}
    
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                
                if not os.path.splitext(image_name)[1].lower() in valid_image_extensions:
                    continue
                
                # Apply each individual augmentation and save the results
                for level in ['weak', 'strong']:
                    # Individual augmentations
                    for aug_type, transform in augmentation_strengths[level].items():
                        augmented_image = augment_image(image_path, [transform])
                        
                        output_dir = os.path.join(output_path, aug_type, level, folder_name)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        save_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_{aug_type}_{level}.png")
                        
                        
                        # Save the augmented image with the specified normalization
                        save_augmented_image(augmented_image, save_path, aug_type)
                        augmented_images[level][aug_type].append(save_path)
                    
                    
    # Stitch together augmented images for each augmentation type and strength
    for level in ['weak', 'strong']:
        for aug_type in augmented_images[level].keys():
            stitched_dir = os.path.join(output_path, 'Stitched', aug_type, level)
            os.makedirs(stitched_dir, exist_ok=True)
            stitched_path = os.path.join(stitched_dir, f"{aug_type}_{level}_stitched.png")
            
            stitch_images(augmented_images[level][aug_type], stitched_path)

# Example usage
dataset_path = '/Users/a86153/Desktop/vlm/brain tumor classification'  # Replace with your dataset path
output_path = '/Users/a86153/Desktop/vlm/brain tumor classification augmented'  # Replace with your output path

process_dataset(dataset_path, output_path)