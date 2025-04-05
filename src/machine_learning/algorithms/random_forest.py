class DecisionTree:
    def __init__(self, max_depth=5):
        """Initialize a decision tree.
        
        Args:
            max_depth (int): Maximum depth of the tree (default: 5).
        """
        self.max_depth = max_depth
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
        if not X or len(set(y)) == 1:  # No split if empty or pure
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
        if depth >= self.max_depth or len(set(y)) == 1 or not X:
            # Return the most common label
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            return max(counts, key=counts.get) if counts else None
        
        feature_idx, threshold, gain = self._best_split(X, y)
        if feature_idx is None or gain == 0.0:
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            return max(counts, key=counts.get)
        
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
        return [self._predict_one(x, self.tree) for x in X]


class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, max_features=None):
        """Initialize the Random Forest model.
        
        Args:
            n_trees (int): Number of trees in the forest (default: 10).
            max_depth (int): Maximum depth of each tree (default: 5).
            max_features (int or None): Number of features to consider per split (default: None, uses sqrt).
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def _sample(self, X, y):
        """Sample data with replacement (bagging).
        
        Args:
            X (list of lists): Input data.
            y (list): Target labels.
        
        Returns:
            tuple: Sampled X and y.
        """
        n_samples = len(X)
        sampled_X, sampled_y = [], []
        for _ in range(n_samples):
            idx = int(n_samples * (sum(i * i for i in X[0]) % 1))  # Simple pseudo-random
            sampled_X.append(X[idx])
            sampled_y.append(y[idx])
        return sampled_X, sampled_y

    def _random_features(self, n_features):
        """Select a random subset of features.
        
        Args:
            n_features (int): Total number of features.
        
        Returns:
            list: Indices of selected features.
        """
        max_f = self.max_features if self.max_features else int(n_features ** 0.5)
        features = list(range(n_features))
        selected = []
        for _ in range(min(max_f, n_features)):
            idx = int(len(features) * (sum(f * f for f in features) % 1))  # Pseudo-random
            selected.append(features.pop(idx))
        return selected

    def fit(self, X, y):
        """Fit the random forest to the data.
        
        Args:
            X (list of lists): Training data.
            y (list): Target labels.
        """
        if not X or not y or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have the same length")
        
        n_features = len(X[0])
        for _ in range(self.n_trees):
            # Bootstrap sample
            X_sample, y_sample = self._sample(X, y)
            # Train a decision tree
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """Predict class labels for new data.
        
        Args:
            X (list of lists): Test data.
        
        Returns:
            list: Predicted class labels.
        """
        if not self.trees:
            raise ValueError("Model must be trained with fit() before predicting")
        
        # Get predictions from all trees
        tree_predictions = [tree.predict(X) for tree in self.trees]
        
        # Majority vote across trees
        predictions = []
        for i in range(len(X)):
            votes = {}
            for tree_pred in tree_predictions:
                label = tree_pred[i]
                votes[label] = votes.get(label, 0) + 1
            predicted_label = max(votes, key=votes.get)
            predictions.append(predicted_label)
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
    y_train = [0, 0, 1, 1]
    
    # Create and fit the model
    rf = RandomForest(n_trees=5, max_depth=3)
    rf.fit(X_train, y_train)
    
    # Test data
    X_test = [
        [3.0, 3.0],  # Near middle
        [1.5, 1.5],  # Near Class 0
        [4.5, 4.5]   # Near Class 1
    ]
    
    # Make predictions
    predictions = rf.predict(X_test)
    
    # Print results
    print("Test samples:", X_test)
    print("Predicted classes:", predictions)