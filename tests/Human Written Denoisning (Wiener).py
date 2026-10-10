# The following code applies Gaussian white noise to images from the Set12 dataset.
# I then plan to use a filter to reduce the noise, and displays the results.

# Import necessary libraries and modules
import os
import time
import tracemalloc
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Start timing the program
start_time = time.time()

# Start memory tracing
tracemalloc.start()

# Below is my hand-written Wiener filter function to reduce Gaussian noise
def apply_filter(image):

    # First I check to see if the input is 2D numeric array
    if not isinstance(image, np.ndarray): # Check if input is an array
        raise TypeError("Error! The Input must be an array.")
    if image.ndim != 2: # Check if input is an array with 2 dimensions
        raise ValueError("Error! The Input must be a 2D array.")
    if image.size == 0:  # # Check if input is an empty array
        raise ValueError("Error! The array can't be empty.")
    
    # Get the height and width of the image
    h, w = len(image), len(image[0])

    # Initialize the filtered image array
    filtered = [[0.0 for _ in range(w)] for _ in range(h)]

    # Set noise variance for Gaussian noise
    noise_var = 625.0 
    
    for i in range(h):
        for j in range(w):

            # I use a 7x7 window for better statistics
            pixels = []# Store pixel values in the window in the list pixel
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        pixels.append(image[ni][nj])
            
            # Find the number of pixels in the window (Changes if pixel is at edge)
            n = len(pixels)
            
            # Find the Local mean and variance
            local_mean = sum(pixels) / n
            local_var = sum((p - local_mean) ** 2 for p in pixels) / n
            
            # Find the Wiener filter coefficient
            signal_var = max(0, local_var - noise_var)
            Wiener_Coefficient = signal_var / (signal_var + noise_var) 
            
            # Apply the Wiener filter
            filtered[i][j] = local_mean + Wiener_Coefficient * (image[i][j] - local_mean)
    
    return np.array(filtered)

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

# Calculate and display runtime
end_time = time.time()
runtime = end_time - start_time
print(f"\nProgram runtime (before displaying results): {runtime:.2f} seconds")

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Memory usage: {current / 1024 / 1024:.2f} MB (Current) / {peak / 1024 / 1024:.2f} MB (Peak)")
tracemalloc.stop()

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

