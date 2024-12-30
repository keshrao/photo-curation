import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from scipy.spatial.distance import hamming

# Step 1: Directory Setup
# Define the directory containing your images.
image_dir = "Images"

# Get a list of all image file paths in the directory.
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]

# Step 2: Understand Perceptual Hashing (pHash)
# pHash generates a hash based on the Discrete Cosine Transform (DCT) of the image.
# It reduces the image to a smaller, simplified version, and compares the resulting patterns.
def compute_phash(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    return imagehash.phash(image)

# Step 3: Understand Wavelet Hashing
# Wavelet hashing generates a hash based on the wavelet transform, capturing image texture and detail.
def compute_wavelet_hash(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    return imagehash.whash(image)

# Step 4: Compute Hashes for Each Image
phashes = {}
whashes = {}
for path in image_paths:
    fname = os.path.basename(path)
    phashes[fname] = compute_phash(path)
    whashes[fname] = compute_wavelet_hash(path)

# Step 5: Compare Hashes
# Use Hamming distance to determine similarity. A smaller Hamming distance means higher similarity.
def compute_hamming_distance(hash1, hash2):
    return hamming(list(hash1.hash.flatten()), list(hash2.hash.flatten()))

similarity_results = []

for i, image1 in enumerate(image_paths):
    for j, image2 in enumerate(image_paths):
        if i < j:  # Avoid duplicate comparisons
            name1, name2 = os.path.basename(image1), os.path.basename(image2)

            phash_distance = compute_hamming_distance(phashes[name1], phashes[name2])
            whash_distance = compute_hamming_distance(whashes[name1], whashes[name2])

            # Calculate weighted sum of distances
            weighted_distance = 0.5 * phash_distance + 0.5 * whash_distance

            similarity_results.append((name1, name2, phash_distance, whash_distance, weighted_distance))

# Step 6: Display Results
# Sort results by weighted distance
similarity_results = sorted(similarity_results, key=lambda x: x[4])

print("Image Similarity Results (Lower Distance = More Similar)")
print("-------------------------------------------------------")
print("Image 1 \t Image 2 \t pHash Distance \t wHash Distance \t Weighted Distance")
for result in similarity_results:
    print(f"{result[0]} \t {result[1]} \t {result[2]:.2f} \t\t {result[3]:.2f} \t\t {result[4]:.2f}")

# Step 7: Interpretation of Results
# Images with small Hamming distances (e.g., < 0.1) are considered similar. Distances close to 1 indicate differences.

# Additional Explanation:
# - pHash focuses on overall structure and brightness patterns.
# - Wavelet Hash focuses on textural and structural details.
# By combining these methods, you can robustly identify similar images even under transformations.
