class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        """Initialize the Decision Tree model.
        
        Args:
            max_depth (int): Maximum depth of the tree (default: 5).
            min_samples_split (int): Minimum number of samples required to split (default: 2).
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini_impurity(self, y):
        """Compute Gini impurity for a list of labels.
        
        Args:
            y (list): Target labels.
        
        Returns:
            float: Gini impurity score.
        """
        if not y:
            return 0.0
        n = len(y)
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        impurity = 1.0
        for count in counts.values():
            impurity -= (count / n) ** 2
        return impurity

    def _split_data(self, X, y, feature_idx, threshold):
        """Split data based on a feature and threshold.
        
        Args:
            X (list of lists): Input data.
            y (list): Target labels.
            feature_idx (int): Index of the feature to split on.
            threshold (float): Threshold value for splitting.
        
        Returns:
            tuple: Left and right splits (X_left, y_left, X_right, y_right).
        """
        X_left, y_left = [], []
        X_right, y_right = [], []
        for i in range(len(X)):
            if X[i][feature_idx] <= threshold:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])
        return X_left, y_left, X_right, y_right

    def _best_split(self, X, y):
        """Find the best feature and threshold to split the data.
        
        Args:
            X (list of lists): Input data.
            y (list): Target labels.
        
        Returns:
            tuple: Best feature index, threshold, and Gini gain.
        """
        if not X or len(set(y)) == 1 or len(X) < self.min_samples_split:
            return None, None, 0.0
        
        n_features = len(X[0])
        parent_impurity = self._gini_impurity(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            values = sorted(set(x[feature_idx] for x in X))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                X_left, y_left, X_right, y_right = self._split_data(X, y, feature_idx, threshold)
                if not y_left or not y_right:
                    continue
                
                n_total = len(y)
                n_left = len(y_left)
                n_right = len(y_right)
                gain = parent_impurity - (n_left / n_total) * self._gini_impurity(y_left) - \
                       (n_right / n_total) * self._gini_impurity(y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree.
        
        Args:
            X (list of lists): Input data.
            y (list): Target labels.
            depth (int): Current depth of the tree.
        
        Returns:
            dict or int/str: Tree node or leaf label.
        """
        # Stopping criteria: max depth, pure node, or too few samples
        if depth >= self.max_depth or len(set(y)) == 1 or len(X) < self.min_samples_split or not X:
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            return max(counts, key=counts.get) if counts else None
        
        # Find the best split
        feature_idx, threshold, gain = self._best_split(X, y)
        if feature_idx is None or gain == 0.0:
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            return max(counts, key=counts.get)
        
        # Split the data and recurse
        X_left, y_left, X_right, y_right = self._split_data(X, y, feature_idx, threshold)
        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1)
        }

    def fit(self, X, y):
        """Fit the decision tree to the data.
        
        Args:
            X (list of lists): Training data.
            y (list): Target labels.
        """
        if not X or not y or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have the same length")
        
        self.tree = self._build_tree(X, y)

    def _predict_one(self, x, node):
        """Predict the class for a single sample.
        
        Args:
            x (list): Test sample.
            node (dict or int/str): Current tree node or leaf.
        
        Returns:
            int/str: Predicted class label.
        """
        if not isinstance(node, dict):
            return node
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        """Predict class labels for new data.
        
        Args:
            X (list of lists): Test data.
        
        Returns:
            list: Predicted class labels.
        """
        if self.tree is None:
            raise ValueError("Model must be trained with fit() before predicting")
        
        return [self._predict_one(x, self.tree) for x in X]


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
    dt = DecisionTree(max_depth=3, min_samples_split=2)
    dt.fit(X_train, y_train)
    
    # Test data
    X_test = [
        [3.0, 3.0],  # Near middle
        [1.5, 1.5],  # Near Class 0
        [4.5, 4.5]   # Near Class 1
    ]
    
    # Make predictions
    predictions = dt.predict(X_test)
    
    # Print results
    print("Test samples:", X_test)
    print("Predicted classes:", predictions)