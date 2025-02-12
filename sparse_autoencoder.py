import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from skimage.metrics import structural_similarity as ssim

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse_loss(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

class SparseAutoencoder:
    def __init__(self, input_size, output_size, hidden_size, sparsity, weight_decay, sparsity_penalty_weight):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        limit = np.sqrt(6 / (input_size + hidden_size))  # Glorot (Xavier) limit
        self.W1 = np.random.uniform(-limit, limit, (hidden_size, input_size))  # Encoder weights
        self.W2 = np.random.uniform(-limit, limit, (output_size, hidden_size))  # Decoder weights
        self.b1 = np.zeros((hidden_size, 1))  # Encoder bias
        self.b2 = np.zeros((output_size, 1))  # Decoder bias
        self.sparsity = sparsity
        self.sparsity_penalty_weight = sparsity_penalty_weight
        self.costs = []

    def forward_pass(self, X):
        """Perform a forward pass."""
        hidden_activation = sigmoid(np.dot(self.W1, X) + self.b1)
        output_activation = sigmoid(np.dot(self.W2, hidden_activation) + self.b2)
        return hidden_activation, output_activation

    def back_propagation(self, X, learning_rate):
        m = X.shape[1]  # Number of training examples

        # Feedforward pass
        hidden_activation, output_activation = self.forward_pass(X)

        # Compute output error
        output_error = -(X - output_activation)
        output_layer = output_error * output_activation * (1 - output_activation)

        # Compute sparsity penalty
        sparsity_hat = (1 / X.shape[1]) * np.sum(hidden_activation, axis=1)
        sparsity_penalty = self.sparsity_penalty_weight * (-(self.sparsity / sparsity_hat) + (1 - self.sparsity) / (1 - sparsity_hat))
        hidden_error = np.dot(self.W2.T, output_layer) + sparsity_penalty.reshape((np.size(sparsity_penalty), 1))
        hidden_layer = hidden_error * hidden_activation * (1 - hidden_activation)

        # Compute gradients
        grad_W1 = np.dot(hidden_layer, X.T) / m + self.weight_decay * self.W1
        grad_b1 = np.sum(hidden_layer, axis=1, keepdims=True) / m
        grad_W2 = np.dot(output_layer, hidden_activation.T) / m + self.weight_decay * self.W2
        grad_b2 = np.sum(output_layer, axis=1, keepdims=True) / m

        # Update weights and biases
        self.W1 -= learning_rate * grad_W1
        self.W2 -= learning_rate * grad_W2
        self.b1 -= learning_rate * grad_b1
        self.b2 -= learning_rate * grad_b2

        # Compute cost
        sum_squared_error = (1 / m) * np.sum(0.5 * (output_error ** 2))
        weight_decay = (self.weight_decay / 2) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        KL = np.sum(self.sparsity * np.log(self.sparsity / sparsity_hat) + (1 - self.sparsity) * np.log((1 - self.sparsity) / (1 - sparsity_hat)))
        cost = sum_squared_error + weight_decay + self.sparsity_penalty_weight * KL
        self.costs.append(cost)

        return cost

    def reconstruct(self, X):
        """Reconstruct the input data after training."""
        _, output_activation = self.forward_pass(X)
        return output_activation


def train_autoencoder(X, autoencoder, epochs, learning_rate, batch_size):
    losses = []  
    for epoch in range(epochs):
        shuffle_indices = np.random.permutation(X.shape[1])
        X_shuffled = X[:, shuffle_indices]
        epoch_loss = 0  
        
        for i in range(0, X.shape[1], batch_size):
            X_batch = X_shuffled[:, i:i + batch_size]
            cost = autoencoder.back_propagation(X_batch, learning_rate)
            _, reconstructed_X_batch = autoencoder.forward_pass(X_batch)
            batch_loss = mse_loss(X_batch, reconstructed_X_batch)
            epoch_loss += batch_loss  # Accumulate loss for the epoch
        
        losses.append(epoch_loss / (X.shape[1] // batch_size))
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, MSE Loss: {losses[-1]:.5f}')
    
    return autoencoder, losses


def train_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)  # Reduce dimensionality
    X_pca_reconstructed = pca.inverse_transform(X_pca)  # Reconstruct data
    return X_pca_reconstructed


def evaluate_classification_accuracy(X, labels, reconstructed_features):
    X_train, X_test, y_train, y_test = train_test_split(reconstructed_features, labels, test_size=0.2, random_state=42)
    classifier = LogisticRegression(max_iter=200)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def compute_ssim(X_original, X_reconstructed):
    num_images = X_original.shape[0]  
    total_ssim = 0
    
    for i in range(num_images):
        original_image = X_original[i].reshape(8, 8)  # Reshape to original image size (8x8 for digits)
        reconstructed_image = X_reconstructed[i].reshape(8, 8)
        ssim_value = ssim(original_image, reconstructed_image, data_range=reconstructed_image.max() - reconstructed_image.min())
        total_ssim += ssim_value
    
    return total_ssim / num_images  



def plot_digit(data, title):
    plt.imshow(data.reshape(8, 8), cmap='gray')
    plt.title(title)
    plt.axis('off')

def plot_reconstructed_images(original, reconstructed, title):
    num_images = 10
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plot_digit(original[i], title="Original")
        plt.subplot(2, num_images, i + 1 + num_images)
        plot_digit(reconstructed[i], title=title)
    plt.tight_layout()
    plt.show()


def plot_2d_scatter(X, labels, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.legend(*scatter.legend_elements(), title="Digits")
    plt.title(title)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.colorbar()
    plt.show()

def visualize_data_quality(autoencoder_features, pca_features, labels):
    tsne_autoencoder = TSNE(n_components=2).fit_transform(autoencoder_features.T)
    tsne_pca = TSNE(n_components=2).fit_transform(pca_features)
    plot_2d_scatter(tsne_autoencoder, labels, "Sparse Autoencoder Feature Space")
    plot_2d_scatter(tsne_pca, labels, "PCA Feature Space")


def compare_autoencoders(X, autoencoder, epochs, learning_rate, batch_size, n_components, labels):
    autoencoder, losses = train_autoencoder(X, autoencoder, epochs, learning_rate, batch_size)
    reconstructed_autoencoder = autoencoder.reconstruct(X)
    reconstructed_pca = train_pca(X.T, n_components)  # Use transpose to match dimensions
    sparse_rmse = rmse_loss(X.T, reconstructed_autoencoder.T)
    pca_rmse = rmse_loss(X.T, reconstructed_pca)
    sparse_accuracy = evaluate_classification_accuracy(X.T, labels, reconstructed_autoencoder.T)
    pca_accuracy = evaluate_classification_accuracy(X.T, labels, reconstructed_pca)
    sparse_ssim = compute_ssim(X.T, reconstructed_autoencoder.T)
    pca_ssim = compute_ssim(X.T, reconstructed_pca)

    print(f'Sparse Autoencoder RMSE: {sparse_rmse:.5f}, Classification Accuracy: {sparse_accuracy:.5f}, SSIM: {sparse_ssim:.5f}')
    print(f'PCA RMSE: {pca_rmse:.5f}, Classification Accuracy: {pca_accuracy:.5f}, SSIM: {pca_ssim:.5f}')

   
    plt.plot(losses)
    plt.title('Sparse Autoencoder MSE Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.show()

    plot_reconstructed_images(X.T, reconstructed_autoencoder.T, title="SAE")
    plot_reconstructed_images(X.T, reconstructed_pca, title="PCA")

def plot_scatter_from_autoencoder(hidden_activations, labels):
   
    tsne = TSNE(n_components=2)
    tsne_activations = tsne.fit_transform(hidden_activations.T)  # Transpose to get (num_samples, hidden_size)

    
    dataframe = pd.DataFrame(data=tsne_activations, columns=["1st component", "2nd component"])
    dataframe['Numbers'] = labels

    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=dataframe, x="1st component", y="2nd component", hue="Numbers", palette='tab10', s=60, alpha=0.7)
    plt.title('t-SNE of Sparse Autoencoder Feature Space')
    plt.show()


def plot_scatter_from_pca(X, labels, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X.T)  # Transpose to get (num_samples, input_size)

    
    dataframe = pd.DataFrame(data=pca_result, columns=["1st component", "2nd component"])
    dataframe['Numbers'] = labels

  
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=dataframe, x="1st component", y="2nd component", hue="Numbers", palette='tab10', s=60, alpha=0.7)
    plt.title('PCA of Feature Space')
    plt.show()


def main():
    digits = load_digits()
    X = digits.data.T  # Shape: (64, 1797) after transpose
    X = X / np.max(X)  # Normalize to [0, 1]

    input_size = X.shape[0]
    output_size = input_size
    hidden_size = 16  # Size of the compressed representation
    sparsity = 0.05
    weight_decay = 0.0001
    sparsity_penalty_weight = 3
    autoencoder = SparseAutoencoder(input_size, output_size, hidden_size, sparsity, weight_decay, sparsity_penalty_weight)
    compare_autoencoders(X, autoencoder, epochs=2000, learning_rate=0.1, batch_size=16, n_components=16, labels=digits.target)
    

    labels = digits.target
    hidden_activations, _ = autoencoder.forward_pass(X)
    plot_scatter_from_autoencoder(hidden_activations, labels)
    plot_scatter_from_pca(X, labels)

main()
