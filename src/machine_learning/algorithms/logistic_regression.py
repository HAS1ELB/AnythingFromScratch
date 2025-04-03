import math  # Using math.log for natural logarithm

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """Initialize the Logistic Regression model.
        
        Args:
            learning_rate (float): Step size for gradient descent (default: 0.01).
            epochs (int): Number of iterations for training (default: 1000).
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None  # Will be initialized during fit
        self.bias = 0.0      # Initial bias term

    def _sigmoid(self, z):
        """Compute the sigmoid activation function.
        
        Args:
            z (float): Input value.
        
        Returns:
            float: Sigmoid output between 0 and 1.
        """
        if z > 100:
            return 1.0
        elif z < -100:
            return 0.0
        return 1.0 / (1.0 + 2.718281828459045 ** (-z))  # e ≈ 2.71828

    def _initialize_parameters(self, n_features):
        """Initialize weights and bias based on the number of features."""
        self.weights = [0.0] * n_features  # One weight per feature

    def _compute_prediction(self, X):
        """Compute predicted probabilities for all samples in X.
        
        Args:
            X (list of lists): Input data, where each inner list is a sample.
        
        Returns:
            list: Predicted probabilities for each sample.
        """
        predictions = []
        for sample in X:
            z = self.bias
            for i in range(len(sample)):
                z += self.weights[i] * sample[i]
            predictions.append(self._sigmoid(z))
        return predictions

    def _compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss.
        
        Args:
            y_true (list): True target values (0 or 1).
            y_pred (list): Predicted probabilities.
        
        Returns:
            float: Average cross-entropy loss.
        """
        n_samples = len(y_true)
        loss = 0.0
        for i in range(n_samples):
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            y_p = max(epsilon, min(1 - epsilon, y_pred[i]))
            loss += - (y_true[i] * math.log(y_p) + (1 - y_true[i]) * math.log(1 - y_p))
        return loss / n_samples

    def _compute_gradients(self, X, y_true, y_pred):
        """Compute gradients for weights and bias.
        
        Args:
            X (list of lists): Input data.
            y_true (list): True target values.
            y_pred (list): Predicted probabilities.
        
        Returns:
            tuple: Gradients for weights and bias.
        """
        n_samples = len(X)
        n_features = len(X[0])
        
        # Initialize gradients
        grad_weights = [0.0] * n_features
        grad_bias = 0.0
        
        # Compute gradients
        for i in range(n_samples):
            error = y_pred[i] - y_true[i]
            for j in range(n_features):
                grad_weights[j] += error * X[i][j]
            grad_bias += error
        
        # Average gradients
        for j in range(n_features):
            grad_weights[j] /= n_samples
        grad_bias /= n_samples
        
        return grad_weights, grad_bias

    def fit(self, X, y):
        """Train the model using gradient descent.
        
        Args:
            X (list of lists): Training data, where each inner list is a sample.
            y (list): Target values (0 or 1).
        """
        if not X or not y or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have the same length")
        
        n_features = len(X[0])
        self._initialize_parameters(n_features)
        
        for epoch in range(self.epochs):
            y_pred = self._compute_prediction(X)
            loss = self._compute_loss(y, y_pred)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
            
            grad_weights, grad_bias = self._compute_gradients(X, y, y_pred)
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * grad_weights[j]
            self.bias -= self.learning_rate * grad_bias

    def predict_proba(self, X):
        """Predict probabilities for new data.
        
        Args:
            X (list of lists): Input data to predict.
        
        Returns:
            list: Predicted probabilities.
        """
        if self.weights is None:
            raise ValueError("Model must be trained with fit() before predicting")
        return self._compute_prediction(X)

    def predict(self, X, threshold=0.5):
        """Predict class labels for new data.
        
        Args:
            X (list of lists): Input data to predict.
            threshold (float): Decision boundary (default: 0.5).
        
        Returns:
            list: Predicted class labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probabilities]


# Example usage
if __name__ == "__main__":
    X_train = [
        [1.0, 1.0],  # Class 0
        [1.5, 0.5],  # Class 0
        [4.0, 4.0],  # Class 1
        [3.5, 4.5]   # Class 1
    ]
    y_train = [0, 0, 1, 1]
    
    model = LogisticRegression(learning_rate=0.01, epochs=1000)
    model.fit(X_train, y_train)
    
    X_test = [
        [2.0, 2.0],
        [5.0, 5.0]
    ]
    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    
    print("\nTrained weights:", model.weights)
    print("Trained bias:", model.bias)
    print("Predicted probabilities:", probabilities)
    print("Predicted classes:", predictions)