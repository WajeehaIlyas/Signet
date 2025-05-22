1. Data Loading & Preprocessing
The script loads image datasets from five language directories (English, Chinese, Hebrew, Urdu, Hindi) using Python’s multiprocessing.Pool for parallel processing. Each image is converted to grayscale, normalized to pixel values between [0, 1], and cropped to extract the most informative 10×10 patch (determined by maximizing local variance). The dataset is then standardized (zero mean, unit variance) and split into 80% training and 20% testing sets using stratified sampling to maintain label distribution.

2. Neural Network Architecture
    - Input Layer: 100 neurons (flattened 10×10 patch).
    - Hidden Layers: Three layers with 512 → 256 → 128 neurons, all using ReLU activation.
    - Output Layer: 5 neurons (one per language) with softmax for classification.
    1. Initialization: He initialization (sqrt(2/n) scaling) for stable training.
    2. Regularization:
        - Dropout (p=0.5) during training to prevent overfitting.
        - L2 Weight Decay (λ=0.001) to penalize large weights.

3. Forward & Backward Pass
    - Forward Pass: Computes activations layer-by-layer with ReLU and dropout (disabled during inference). The final output is a probability distribution via softmax.
    - Backward Pass: Implements cross-entropy loss with gradients computed for:
        1. Weight updates using mini-batch gradient descent.
        2. L2 regularization to shrink large weights.
        3. Proper handling of ReLU gradients (zero for non-positive inputs).

4. Training Loop
    - Shuffles data each epoch.
    - Processes batches (batch_size=64) for memory efficiency.
    - Tracks best weights and restores them if no improvement occurs for 5 epochs (early stopping).
    - Logs training/test accuracy and loss per epoch.
    - Uses a fixed learning rate (0.005) without scheduling.

5. Evaluation & Inference
    - Accuracy Calculation: Compares predicted vs. true labels.
    - Loss Computation: Cross-entropy loss with a small epsilon (1e-10) to avoid log(0).
    - Classification Pipeline (classify_image):
        1. Loads and preprocesses an image.
        2. Extracts the 10×10 patch.
        3. Normalizes using global mean/std.
        4. Runs forward pass and decodes the predicted label.

6. Example Usage
    - Training with early stopping.
    - Test set evaluation (accuracy and loss).
    - Batch prediction on example images (e.g., test_images/image1.png).

7. Key Strengths
    - Manual Implementation: No reliance on high-level frameworks (e.g., TensorFlow/PyTorch).
    - Efficiency: Multiprocessing for fast data loading.
    - Robustness: Dropout + L2 regularization to combat overfitting.
    - Reproducibility: Fixed random seed (np.random.seed(42)).

8. Requirements
- Python 3.6+
- Required packages:
- numpy
- Pillow (PIL)
- scikit-learn