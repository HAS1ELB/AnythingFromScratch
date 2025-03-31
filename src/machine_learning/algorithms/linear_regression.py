class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """Initialize the Linear Regression model.
        
        Args:
            learning_rate (float): Step size for gradient descent (default: 0.01).
            epochs (int): Number of iterations for training (default: 1000).
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None  # Will be initialized during fit
        self.bias = 0.0      # Initial bias term

    def _initialize_parameters(self, n_features):
        """Initialize weights and bias based on the number of features."""
        self.weights = [0.0] * n_features  # One weight per feature

    def _compute_prediction(self, X):
        """Compute predictions for all samples in X.
        
        Args:
            X (list of lists): Input data, where each inner list is a sample.
        
        Returns:
            list: Predicted values for each sample.
        """
        predictions = []
        for sample in X:
            pred = self.bias
            for i in range(len(sample)):
                pred += self.weights[i] * sample[i]
            predictions.append(pred)
        return predictions

    def _compute_loss(self, y_true, y_pred):
        """Compute Mean Squared Error (MSE) loss.
        
        Args:
            y_true (list): True target values.
            y_pred (list): Predicted values.
        
        Returns:
            float: MSE loss.
        """
        n_samples = len(y_true)
        loss = 0.0
        for i in range(n_samples):
            loss += (y_true[i] - y_pred[i]) ** 2
        return loss / n_samples

    def _compute_gradients(self, X, y_true, y_pred):
        """Compute gradients for weights and bias.
        
        Args:
            X (list of lists): Input data.
            y_true (list): True target values.
            y_pred (list): Predicted values.
        
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
            y (list): Target values.
        """
        # Validate input
        if not X or not y or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have the same length")
        
        n_features = len(X[0])
        self._initialize_parameters(n_features)
        
        # Training loop
        for epoch in range(self.epochs):
            # Forward pass: compute predictions
            y_pred = self._compute_prediction(X)
            
            # Compute loss (for monitoring, optional)
            loss = self._compute_loss(y, y_pred)
            if epoch % 100 == 0:  # Print loss every 100 epochs
                print(f"Epoch {epoch}, Loss: {loss}")
            
            # Backward pass: compute gradients
            grad_weights, grad_bias = self._compute_gradients(X, y, y_pred)
            
            # Update parameters
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * grad_weights[j]
            self.bias -= self.learning_rate * grad_bias

    def predict(self, X):
        """Make predictions for new data.
        
        Args:
            X (list of lists): Input data to predict.
        
        Returns:
            list: Predicted values.
        """
        if self.weights is None:
            raise ValueError("Model must be trained with fit() before predicting")
        return self._compute_prediction(X)


# Example usage
if __name__ == "__main__":
    # Synthetic data: y = 2 * x1 + 3 * x2 + 1 + noise
    X_train = [
        [1.0, 2.0],  # Sample 1
        [2.0, 3.0],  # Sample 2
        [3.0, 1.0],  # Sample 3
        [4.0, 4.0]   # Sample 4
    ]
    y_train = [7.1, 11.2, 9.0, 17.3]  # Target values with slight noise
    
    # Create and train the model
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    X_test = [
        [5.0, 2.0],  # Test sample 1
        [1.5, 3.5]   # Test sample 2
    ]
    predictions = model.predict(X_test)
    
    # Print results
    print("\nTrained weights:", model.weights)
    print("Trained bias:", model.bias)
    print("Predictions:", predictions)