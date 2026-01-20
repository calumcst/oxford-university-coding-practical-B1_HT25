# The following code applies Gaussian white noise to images from the Set12 dataset.
# I then plan to use a filter to reduce the noise, and displays the results.

# Import necessary libraries and modules
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define filter function to reduce the noise of an grayscale image
def apply_filter(image):
    # image: 2D list of pixel intensities (0-255)
    height = len(image)
    width = len(image[0])
    output = [[0 for _ in range(width)] for _ in range(height)]

    # 3x3 mean filter kernel
    for i in range(height):
        for j in range(width):
            total = 0
            count = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        total += image[ni][nj]
                        count += 1
            output[i][j] = int(round(total / count))
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
    psnr_filtered = calculate_psnr(original, filtered_image)
    
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

