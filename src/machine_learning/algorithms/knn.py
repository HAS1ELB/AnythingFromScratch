class KNN:
    def __init__(self, k=3):
        """Initialize the k-Nearest Neighbors model.
        
        Args:
            k (int): Number of neighbors to consider (default: 3).
        """
        self.k = k
        self.X_train = None  # Training data
        self.y_train = None  # Training labels

    def _euclidean_distance(self, x1, x2):
        """Compute the Euclidean distance between two points.
        
        Args:
            x1 (list): First point.
            x2 (list): Second point.
        
        Returns:
            float: Euclidean distance.
        """
        if len(x1) != len(x2):
            raise ValueError("Feature dimensions must match")
        
        distance = 0.0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i]) ** 2
        return distance ** 0.5

    def fit(self, X, y):
        """Store the training data and labels.
        
        Args:
            X (list of lists): Training data, where each inner list is a sample.
            y (list): Target labels.
        """
        if not X or not y or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have the same length")
        
        self.X_train = X
        self.y_train = y

    def _get_neighbors(self, x_test):
        """Find the k nearest neighbors to a test sample.
        
        Args:
            x_test (list): Test sample.
        
        Returns:
            list: Indices of the k nearest neighbors.
        """
        distances = []
        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(x_test, self.X_train[i])
            distances.append((dist, i))
        
        # Sort by distance and get top k
        distances.sort(key=lambda x: x[0])
        neighbors = [distances[i][1] for i in range(self.k)]
        return neighbors

    def _majority_vote(self, neighbor_indices):
        """Determine the predicted class by majority vote.
        
        Args:
            neighbor_indices (list): Indices of the k nearest neighbors.
        
        Returns:
            int/str: Predicted class label.
        """
        votes = {}
        for idx in neighbor_indices:
            label = self.y_train[idx]
            votes[label] = votes.get(label, 0) + 1
        
        # Find the label with the most votes
        max_votes = 0
        predicted_label = None
        for label, count in votes.items():
            if count > max_votes:
                max_votes = count
                predicted_label = label
        return predicted_label

    def predict(self, X):
        """Predict class labels for new data.
        
        Args:
            X (list of lists): Test data to predict.
        
        Returns:
            list: Predicted class labels.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be trained with fit() before predicting")
        
        predictions = []
        for x_test in X:
            neighbors = self._get_neighbors(x_test)
            prediction = self._majority_vote(neighbors)
            predictions.append(prediction)
        return predictions


# Example usage
if __name__ == "__main__":
    # Synthetic data: 2D points with two classes
    X_train = [
        [1.0, 2.0],  # Class 0
        [2.0, 1.0],  # Class 0
        [4.0, 5.0],  # Class 1
        [5.0, 4.0]   # Class 1
    ]
    y_train = [0, 0, 1, 1]  # Target labels
    
    # Create and fit the model
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    
    # Test data
    X_test = [
        [3.0, 3.0],  # Near middle
        [1.5, 1.5],  # Near Class 0
        [4.5, 4.5]   # Near Class 1
    ]
    
    # Make predictions
    predictions = knn.predict(X_test)
    
    # Print results
    print("Test samples:", X_test)
    print("Predicted classes:", predictions)