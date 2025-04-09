# CNN Encoding Analysis and Clustering

## Overview
This project analyzes encodings extracted from the penultimate fully connected layer of a Convolutional Neural Network (CNN) model, selected for its strong performance on a test dataset. These encodings serve as compact representations of the data, which are then visualized and clustered using PCA, K-Means, DBSCAN, and t-SNE. The analysis explores cluster quality, class separation, and potential subcategories within classes, using the Fashion MNIST dataset (implied by class labels like "Sneakers," "Trousers"). Random image sampling from clusters provides insights into the semantic content of the encodings.

## Methodology
The CNN encodings were processed and evaluated as follows:

### 1. PCA Visualization
- **Approach**: Applied Principal Component Analysis (PCA) to reduce encodings to 2D, using the first two components.
- **Visualization**: Plotted with original class labels as color mappings.

### 2. K-Means Clustering
- **Approach**: Performed K-Means clustering on the encodings (number of clusters not specified, assumed 5 based on analysis).
- **Visualization**: Plotted with cluster labels as color mappings.
- **Analysis**: Sampled 10 random images per cluster to infer content.

### 3. DBSCAN Clustering
- **Approach**: Applied Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to the encodings.
- **Visualization**: Plotted with cluster labels as color mappings.

### 4. t-SNE Visualization
- **Approach**: Used t-Distributed Stochastic Neighbor Embedding (t-SNE) to reduce encodings to 2D.
- **Visualization**: Plotted with original class labels as color mappings.
- **Analysis**: Sampled 10 random images from distinct "blobs" within each class to explore subcategories.

## Results

### PCA Visualization
- **Observation**: Distinct clusters emerge for each class, with some overlap at boundaries. Class 4 (yellow) shows the most dispersed and overlapping distribution.

### K-Means Clustering
- **Observation**: Produces distinct clusters with minor boundary overlap.
- **Cluster Analysis** (10 random images per cluster):
  - **Cluster 0**: Pants (all labeled [2]) → Likely "Trousers."
  - **Cluster 1**: Bags and sneakers (all labeled [3]) → Mixed "Bags/Sneakers."
  - **Cluster 2**: Sandals (all labeled [1]) → Likely "Sandals."
  - **Cluster 3**: Mixed upperwear (labels [0], [2], [4]) → "T-shirts, Pullovers, Dresses, Coats."
  - **Cluster 4**: Ankle boots (all labeled [4]) → Likely "Ankle Boots."

### DBSCAN Clustering
- **Observation**: Fails to define distinct clusters, assigning most data to a single cluster.
- **Reason**: DBSCAN struggles with uniform-density feature spaces, as seen in the encodings.

### t-SNE Visualization
- **Observation**: Reveals distinct clusters per class, with some classes (e.g., Class 3) forming multiple blobs.
- **Subcategory Analysis** (10 random images per blob):
  - **Class 0 (Purple)**: T-shirts, Shirts, Tops (women’s).
  - **Class 1 (Dark Blue)**: Sandals (flat), Sandals (high heels).
  - **Class 2 (Turquoise)**: Trousers, Pullovers (two blobs).
  - **Class 3 (Green)**: Sneakers, Bags, Purses (three blobs).
  - **Class 4 (Yellow)**: Ankle boots, Dresses, Full-sleeve Dresses, Coats (four blobs).
- **Proposed Labels**:
  - Class 0: T-shirts, Shirts, Tops.
  - Class 1: Sandals.
  - Class 2: Trousers, Pullovers.
  - Class 3: Sneakers, Bags.
  - Class 4: Ankle Boots, Dresses, Coats.

## Insights
- **Encoding Quality**: Distinct clusters in PCA and t-SNE confirm the CNN encodings effectively capture data variations.
- **Clustering Performance**:
  - K-Means excels at defining compact clusters, though some (e.g., Cluster 3) mix labels with semantic similarity (upperwear).
  - DBSCAN underperforms due to uniform density in the feature space.
  - t-SNE highlights fine-grained subcategories (e.g., Class 3’s Sneakers vs. Bags), suggesting encodings preserve nuanced differences.
- **Class Complexity**: Classes like 3 and 4 show multiple blobs, indicating diverse object types within a single label.

## Requirements
- **Python**: Core language.
- **PyTorch**: CNN model and encoding extraction (implied).
- **NumPy**: Data handling.
- **Scikit-learn**: PCA, K-Means, DBSCAN implementations.
- **Matplotlib**: Visualization of clusters.
- **t-SNE**: Via `scikit-learn` or `openTSNE`.

## How to Run
1. **Extract Encodings**: Use the best-performing CNN model to generate encodings from the test dataset’s penultimate layer.
2. **Visualize with PCA**: Reduce to 2D, plot with class labels.
3. **Cluster with K-Means**: Apply K-Means, plot clusters, sample 10 images per cluster.
4. **Cluster with DBSCAN**: Apply DBSCAN, plot results.
5. **Visualize with t-SNE**: Reduce to 2D, plot with class labels, sample 10 images per blob.
6. **Analyze**: Compare cluster labels to original classes and inspect images.

## Future Work
- Test additional clustering methods (e.g., hierarchical clustering).
- Adjust DBSCAN parameters (eps, min_samples) for better separation.
- Explore encodings from earlier CNN layers for comparison.
