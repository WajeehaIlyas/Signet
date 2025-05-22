import os
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool

# Dataset directories
ENGLISH_DIR = '/home/wajeeha/Documents/classifier2/Dataset/english'
CHINESE_DIR = '/home/wajeeha/Documents/classifier2/Dataset/chinese'
HEBREW_DIR = '/home/wajeeha/Documents/classifier2/Dataset/hebrew'
URDU_DIR = '/home/wajeeha/Documents/classifier2/Dataset/urdu'
HINDI_DIR = '/home/wajeeha/Documents/classifier2/Dataset/hindi'

CROP_SIZE = 10

def normalize_dataset(X):
    #Normalize the dataset to have zero mean and unit variance
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-7  
    return (X - mean) / std, mean, std

def crop_relevant_patch(img_array, patch_size=CROP_SIZE):
    #Crop a region from image where strokes (variance) are highest
    from numpy.lib.stride_tricks import sliding_window_view
    
    h, w = img_array.shape
    #if image is smaller than patch size, return zero array
    if h < patch_size or w < patch_size:
        return np.zeros((patch_size, patch_size))
    
    # Calculate the variance of each patch
    windows = sliding_window_view(img_array, (patch_size, patch_size))
    variances = np.var(windows, axis=(2, 3))
    
    # Find the patch with the maximum variance
    max_idx = np.argmax(variances)
    max_pos = np.unravel_index(max_idx, variances.shape)
    
    return windows[max_pos]

def process_single_image(filepath, label):
    #Process a single image
    try:
        #convert to grey scale and normalize
        img = Image.open(filepath).convert('L')                
        img_array = np.array(img).astype(np.float32) / 255.0
        patch = crop_relevant_patch(img_array, CROP_SIZE)
        #if image extracted, return ot along with its label
        if patch is not None:
            return patch.flatten(), label
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return None, None

def load_images_from_folder(folder, label):
    #Load images from folder using multiprocessing
    images, labels = [], []
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    with Pool() as pool:
        results = pool.starmap(process_single_image, 
                             [(fp, label) for fp in filepaths])
    
    for img_data, lbl in results:
        if img_data is not None:
            images.append(img_data)
            labels.append(lbl)
    
    return images, labels

def load_all_datasets():
    #Load all datasets
    print("Loading datasets...")
    all_data = []
    all_labels = []
    
    for path, label in [
        (ENGLISH_DIR, 'english'),
        (CHINESE_DIR, 'chinese'),
        (HEBREW_DIR, 'hebrew'),
        (URDU_DIR, 'urdu'),
        (HINDI_DIR, 'hindi')
    ]:
        data, labels = load_images_from_folder(path, label)
        all_data.extend(data)
        all_labels.extend(labels)
        print(f"Loaded {len(data)} samples from {label}")
    
    return np.array(all_data), np.array(all_labels)

# Load and preprocess all data
X_raw, y_raw = load_all_datasets()

# Normalize the dataset
print("Normalizing data...")
X_normalized, global_mean, global_std = normalize_dataset(X_raw)

#convert string labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42, stratify=y
)

# Model parameters
input_size = CROP_SIZE * CROP_SIZE
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 128
output_size = len(label_encoder.classes_)

# He initialization
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
b1 = np.zeros(hidden_size1)
W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
b2 = np.zeros(hidden_size2)
W3 = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(2. / hidden_size2)
b3 = np.zeros(hidden_size3)
W4 = np.random.randn(hidden_size3, output_size) * np.sqrt(2. / hidden_size3)
b4 = np.zeros(output_size)

# Activations
#convert negative values to zero
def relu(x): return np.maximum(0, x)

#values grater than 0 => 1
def relu_derivative(x): return (x > 0).astype(np.float32)

#probabilities of each class
def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def forward(X, dropout_rate=0.5, training=True):
    #input to hidden layer 1
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    #randomly drop neurons to prevent overfitting
    if training and dropout_rate > 0:
        mask = (np.random.rand(*a1.shape) > dropout_rate) / (1 - dropout_rate)
        a1 *= mask

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    if training and dropout_rate > 0:
        mask = (np.random.rand(*a2.shape) > dropout_rate) / (1 - dropout_rate)
        a2 *= mask

    z3 = np.dot(a2, W3) + b3
    a3 = relu(z3)
    if training and dropout_rate > 0:
        mask = (np.random.rand(*a3.shape) > dropout_rate) / (1 - dropout_rate)
        a3 *= mask

    z4 = np.dot(a3, W4) + b4
    output = softmax(z4)
    return z1, a1, z2, a2, z3, a3, z4, output

def backward(X, y, z1, a1, z2, a2, z3, a3, z4, output, learning_rate=0.001, l2_lambda=0.001):
    global W1, b1, W2, b2, W3, b3, W4, b4
    m = X.shape[0]

    #zero matrix with same shape as output
    y_one_hot = np.zeros_like(output)
    #set the correct class to 1
    y_one_hot[np.arange(m), y] = 1

#gradient of loss w.r.t. output
    dz4 = (output - y_one_hot) / m
    dW4 = np.dot(a3.T, dz4) + (l2_lambda * W4) / m
    #gradient of loss w.r.t. bias
    db4 = np.sum(dz4, axis=0)

#gradient wrt activations of layer 3
    da3 = np.dot(dz4, W4.T)
    #ensure that only active neurons propagate the gradient
    dz3 = da3 * relu_derivative(z3)
    dW3 = np.dot(a2.T, dz3) + (l2_lambda * W3) / m
    db3 = np.sum(dz3, axis=0)

    da2 = np.dot(dz3, W3.T)
    dz2 = da2 * relu_derivative(z2)
    dW2 = np.dot(a1.T, dz2) + (l2_lambda * W2) / m
    db2 = np.sum(dz2, axis=0)

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(X.T, dz1) + (l2_lambda * W1) / m
    db1 = np.sum(dz1, axis=0)

    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

def predict(X):
    _, _, _, _, _, _, _, output = forward(X, dropout_rate=0.0, training=False)
    return np.argmax(output, axis=1)

def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def classify_image(image_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img).astype(np.float32) / 255.0
    patch = crop_relevant_patch(img_array)
    normalized = (patch.flatten() - global_mean) / global_std
    X = normalized.reshape(1, -1)
    pred_idx = predict(X)[0]
    return label_encoder.inverse_transform([pred_idx])[0]

def evaluate(X, y):
    preds = predict(X)
    acc = accuracy(preds, y)
    print(f"Accuracy: {acc:.4f}")
    return acc

def compute_loss(X, y):
    #Compute cross-entropy loss
    _, _, _, _, _, _, _, output = forward(X, dropout_rate=0.0, training=False)
    m = y.shape[0]
    y_one_hot = np.zeros_like(output)
    y_one_hot[np.arange(m), y] = 1
    loss = -np.mean(y_one_hot * np.log(output + 1e-10))  
    return loss

def train_model(X_train, y_train, X_test, y_test, epochs=100, initial_lr=0.1, batch_size=64, patience=2):
    best_test_acc = 0.0
    best_weights = None
    no_improvement = 0
    lr = initial_lr
    
    # Store original weights for restoration
    original_weights = {
        'W1': W1.copy(), 'b1': b1.copy(),
        'W2': W2.copy(), 'b2': b2.copy(),
        'W3': W3.copy(), 'b3': b3.copy(),
        'W4': W4.copy(), 'b4': b4.copy()
    }

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Training loop
        for i in range(0, len(X_train_shuffled), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            z1, a1, z2, a2, z3, a3, z4, output = forward(X_batch, training=True)
            backward(X_batch, y_batch, z1, a1, z2, a2, z3, a3, z4, output, learning_rate=lr)

        # Calculate metrics
        train_acc = accuracy(predict(X_train), y_train)
        test_acc = accuracy(predict(X_test), y_test)
        train_loss = compute_loss(X_train, y_train)
        test_loss = compute_loss(X_test, y_test)
        
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Early stopping logic
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            no_improvement = 0
            # Save best weights
            best_weights = {
                'W1': W1.copy(), 'b1': b1.copy(),
                'W2': W2.copy(), 'b2': b2.copy(),
                'W3': W3.copy(), 'b3': b3.copy(),
                'W4': W4.copy(), 'b4': b4.copy()
            }
            print("  New best test accuracy!")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"  Early stopping after {patience} epochs without improvement")
                # Restore best weights
                W1[:], b1[:] = best_weights['W1'], best_weights['b1']
                W2[:], b2[:] = best_weights['W2'], best_weights['b2']
                W3[:], b3[:] = best_weights['W3'], best_weights['b3']
                W4[:], b4[:] = best_weights['W4'], best_weights['b4']
                break

    # If we didn't early stop but completed all epochs, ensure we have best weights
    if best_weights is not None and no_improvement < patience:
        W1[:], b1[:] = best_weights['W1'], best_weights['b1']
        W2[:], b2[:] = best_weights['W2'], best_weights['b2']
        W3[:], b3[:] = best_weights['W3'], best_weights['b3']
        W4[:], b4[:] = best_weights['W4'], best_weights['b4']

    return best_test_acc

# Train the model with early stopping
print("Starting training with early stopping...")
best_acc = train_model(
    X_train, y_train, X_test, y_test,
    epochs=2000,
    initial_lr=0.1,
    batch_size=64,
    patience=5  # Stop if no improvement for 5 epochs
)

print(f"\nBest test accuracy: {best_acc:.4f}")

# Evaluate on test set
print("\nFinal evaluation on test set:")
test_acc = accuracy(predict(X_test), y_test)
test_loss = compute_loss(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Evaluate on test set
print("\nFinal evaluation on test set:")
test_acc = evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Test on multiple images
def test_on_multiple_images(image_paths):
    for image_path in image_paths:
        predicted_label = classify_image(image_path)
        print(f"Predicted label for {image_path}: {predicted_label}")

# Example usage
image_paths = [
    '/home/wajeeha/Documents/classifier2/test_images/image1.png',
    '/home/wajeeha/Documents/classifier2/test_images/image2.png',
    '/home/wajeeha/Documents/classifier2/test_images/image3.png',
    '/home/wajeeha/Documents/classifier2/test_images/image4.png',
    '/home/wajeeha/Documents/classifier2/test_images/image5.png'
]

test_on_multiple_images(image_paths)