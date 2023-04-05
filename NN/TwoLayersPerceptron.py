from math import ceil
import numpy as np

class TwoLayerPerceptron:
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        # Initialize weights and biases with random values
        self.W1 = np.random.randn(n_input, n_hidden)
        self.b1 = np.random.randn(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_output)
        self.b2 = np.random.randn(n_output)
        
    def forward(self, X):
        # Compute activations of hidden layer
        hidden_activations = np.dot(X, self.W1) + self.b1
        hidden_output = self.sigmoid(hidden_activations)
        # Compute activations of output layer
        output_activations = np.dot(hidden_output, self.W2) + self.b2
        predicted_output = self.sigmoid(output_activations)
        predicted_label = np.argmax(predicted_output, axis=1)
        return predicted_label
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def backpropagation(self, X, y, output):
        # Compute error at output layer
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        # Compute error at hidden layer
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(X)
        # Update weights and biases
        self.W2 += self.learning_rate * np.dot(X.T, output_delta)
        self.b2 += self.learning_rate * np.sum(output_delta, axis=0)
        self.W1 += self.learning_rate * np.dot(X.T, hidden_delta)
        self.b1 += self.learning_rate * np.sum(hidden_delta, axis=0)
    
    def train(self, X, y, n_epochs=5):
        np.random.seed(42)
        idx = np.random.randint(len(y))
        train_idx, val_idx = idx[:ceil(0.8*len(idx))], idx[ceil(0.8*len(idx)):]
        train_X, train_y = X[train_idx], y[train_idx]
        val_X, val_y = X[val_idx], y[val_idx]

        for i in range(n_epochs):
            ### training
            # Forward pass
            output = self.forward(train_X)
            # Backward pass
            self.backpropagation(train_X, train_y, output)

            ### validation
            # Forward pass
            val_output = self.forward(val_X)
            
            # Compute and print loss every 100 epochs
            if i % 100 == 0:
                train_loss = np.mean(np.square(train_y - output))
                val_loss = np.mean(np.square(val_y - output))
                print(f"Epoch {i} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")