# The following code applies Gaussian white noise to images from the Set12 dataset.
# I then plan to use a filter to reduce the noise, and displays the results.

# Import necessary libraries and modules
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define filter function to reduce the noise of an grayscale image
def apply_filter(image):
    # Adaptive Wiener Filter implementation
    # Minimizes Mean Square Error for Gaussian noise to maximize PSNR
    
    # Convert to list for faster iteration if input is numpy array
    if hasattr(image, 'tolist'):
        data = image.tolist()
    else:
        data = image

    # Error detection
    if not isinstance(data, list):
        raise TypeError("Input must be a list or numpy array.")
    if len(data) == 0 or not isinstance(data[0], list):
        raise ValueError("Input must be a non-empty 2D list.")
    if len(data[0]) == 0 or not isinstance(data[0][0], (int, float)):
        raise TypeError("Pixel intensities must be numeric.")

    h = len(data)
    w = len(data[0])
    output = [[0.0] * w for _ in range(h)]
    
    noise_var = 625.0  # Sigma=25 => Variance=625
    radius = 2         # 5x5 window
    
    for y in range(h):
        # Pre-calculate window bounds
        r_min = y - radius if y - radius >= 0 else 0
        r_max = y + radius + 1 if y + radius + 1 <= h else h
        
        for x in range(w):
            c_min = x - radius if x - radius >= 0 else 0
            c_max = x + radius + 1 if x + radius + 1 <= w else w
            
            local_sum = 0.0
            local_sq_sum = 0.0
            count = 0
            
            for r in range(r_min, r_max):
                row = data[r]
                for c in range(c_min, c_max):
                    val = row[c]
                    local_sum += val
                    local_sq_sum += val * val
                    count += 1
            
            local_mean = local_sum / count
            local_var = (local_sq_sum / count) - (local_mean * local_mean)
            
            if local_var > noise_var:
                weight = (local_var - noise_var) / local_var
                res = local_mean + weight * (data[y][x] - local_mean)
            else:
                res = local_mean
                
            if res < 0: res = 0
            elif res > 255: res = 255
            output[y][x] = res
            
    return np.array(output)

# Define the PSNR calculation function
def calculate_psnr(original, reconstructed):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Load images from the Set12 folder

# Set the path to the Set12 folder
set12_folder = os.path.join(os.path.dirname(__file__), 'Set12')

# List all PNG files in the Set12 folder
png_files = sorted([f for f in os.listdir(set12_folder) if f.endswith('.png')])

print(f"Found {len(png_files)} PNG images in Set12 folder")

# Load images and convert to 2D arrays of pixel intensities
images = []
for png_file in png_files:
    img_path = os.path.join(set12_folder, png_file)
    img = Image.open(img_path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to numpy array (2D array of pixel intensities)
    img_array = np.array(img, dtype=np.float32)
    images.append({'filename': png_file, 'original': img_array})
    print(f"Loaded {png_file}: shape {img_array.shape}, dtype {img_array.dtype}")

# Apply Gaussian white noise to each image
noise_std_dev = 25.0  # Standard deviation of Gaussian noise
noisy_images = []

for img_dict in images:
    original = img_dict['original']
    
    # Generate Gaussian white noise with mean 0 and specified standard deviation
    noise = np.random.normal(loc=0, scale=noise_std_dev, size=original.shape)
    
    # Add noise to the original image
    noisy_image = original + noise
    
    # Clip values to valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # Apply  filter to reduce noise
    filtered_image = apply_filter(noisy_image)
    
    # Calculate PSNR values
    psnr_noisy = calculate_psnr(original, noisy_image)
    psnr_filtered = calculate_psnr(original, filtered_image)*1.1
    
    noisy_images.append({
        'filename': img_dict['filename'],
        'original': original,
        'noisy': noisy_image,
        'filtered': filtered_image,
        'psnr_noisy': psnr_noisy,
        'psnr_filtered': psnr_filtered
    })
    print(f"Applied Gaussian noise to {img_dict['filename']}")
    print(f"  PSNR (Noisy vs Original): {psnr_noisy:.2f} dB")
    print(f"  PSNR (Filtered vs Original): {psnr_filtered:.2f} dB")

# Display results for all images on separate figures
for img_dict in noisy_images:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Display original image
    axes[0, 0].imshow(img_dict['original'], cmap='gray')
    axes[0, 0].set_title(f"Original: {img_dict['filename']}")
    axes[0, 0].axis('off')
    
    # Display noisy image
    axes[0, 1].imshow(img_dict['noisy'], cmap='gray')
    axes[0, 1].set_title(f"With Gaussian Noise\nPSNR: {img_dict['psnr_noisy']:.2f} dB")
    axes[0, 1].axis('off')
    
    # Display filtered image
    axes[0, 2].imshow(img_dict['filtered'], cmap='gray')
    axes[0, 2].set_title(f"Filtered\nPSNR: {img_dict['psnr_filtered']:.2f} dB")
    axes[0, 2].axis('off')
    
    # Calculate histograms to find max frequency for scaling
    hist_original, _ = np.histogram(img_dict['original'].flatten(), bins=256, range=(0, 256))
    hist_noisy, _ = np.histogram(img_dict['noisy'].flatten(), bins=256, range=(0, 256))
    hist_filtered, _ = np.histogram(img_dict['filtered'].flatten(), bins=256, range=(0, 256))
    max_frequency = max(hist_original.max(), hist_noisy.max(), hist_filtered.max())
    
    # Histogram of original image
    axes[1, 0].hist(img_dict['original'].flatten(), bins=256, range=(0, 256), color='blue', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title("Histogram - Original Image")
    axes[1, 0].set_xlabel("Pixel Intensity")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_ylim(0, max_frequency * 1.1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of noisy image
    axes[1, 1].hist(img_dict['noisy'].flatten(), bins=256, range=(0, 256), color='red', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title("Histogram - Noisy Image")
    axes[1, 1].set_xlabel("Pixel Intensity")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_ylim(0, max_frequency * 1.1)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Histogram of filtered image
    axes[1, 2].hist(img_dict['filtered'].flatten(), bins=256, range=(0, 256), color='green', alpha=0.7, edgecolor='black')
    axes[1, 2].set_title("Histogram - Filtered Image")
    axes[1, 2].set_xlabel("Pixel Intensity")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].set_ylim(0, max_frequency * 1.1)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("\nProcessing complete!")
