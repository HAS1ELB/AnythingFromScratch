class PCA:
    def __init__(self, n_components):
        """Initialize the PCA model.
        
        Args:
            n_components (int): Number of principal components to keep.
        """
        self.n_components = n_components
        self.components = None  # Principal components (eigenvectors)
        self.mean = None        # Mean of the data

    def _mean_center(self, X):
        """Center the data by subtracting the mean of each feature.
        
        Args:
            X (list of lists): Input data, where each inner list is a sample.
        
        Returns:
            list of lists: Mean-centered data.
        """
        n_samples = len(X)
        n_features = len(X[0])
        self.mean = [0.0] * n_features
        
        # Compute mean for each feature
        for j in range(n_features):
            for i in range(n_samples):
                self.mean[j] += X[i][j]
            self.mean[j] /= n_samples
        
        # Subtract mean from each sample
        centered_X = []
        for i in range(n_samples):
            centered_sample = [X[i][j] - self.mean[j] for j in range(n_features)]
            centered_X.append(centered_sample)
        return centered_X

    def _matrix_multiply(self, A, B):
        """Multiply two matrices A and B.
        
        Args:
            A (list of lists): First matrix.
            B (list of lists): Second matrix.
        
        Returns:
            list of lists: Resulting matrix.
        """
        rows_A = len(A)
        cols_A = len(A[0])
        cols_B = len(B[0])
        result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
        
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    def _transpose(self, X):
        """Transpose a matrix.
        
        Args:
            X (list of lists): Input matrix.
        
        Returns:
            list of lists: Transposed matrix.
        """
        n_samples = len(X)
        n_features = len(X[0])
        return [[X[i][j] for i in range(n_samples)] for j in range(n_features)]

    def _covariance_matrix(self, X):
        """Compute the covariance matrix of the data.
        
        Args:
            X (list of lists): Mean-centered data.
        
        Returns:
            list of lists: Covariance matrix.
        """
        n_samples = len(X)
        X_T = self._transpose(X)
        cov = self._matrix_multiply(X_T, X)
        
        # Divide by (n_samples - 1) for unbiased estimate
        for i in range(len(cov)):
            for j in range(len(cov[0])):
                cov[i][j] /= (n_samples - 1)
        return cov

    def _power_iteration(self, A, max_iter=1000, tol=1e-6):
        """Compute the dominant eigenvector using power iteration.
        
        Args:
            A (list of lists): Input matrix (covariance matrix).
            max_iter (int): Maximum iterations.
            tol (float): Convergence tolerance.
        
        Returns:
            list: Dominant eigenvector.
        """
        n = len(A)
        # Initialize random vector
        v = [1.0] * n
        norm = sum(x * x for x in v) ** 0.5
        v = [x / norm for x in v]
        
        for _ in range(max_iter):
            # Matrix-vector multiplication
            v_new = [0.0] * n
            for i in range(n):
                for j in range(n):
                    v_new[i] += A[i][j] * v[j]
            
            # Normalize
            norm = sum(x * x for x in v_new) ** 0.5
            v_new = [x / norm for x in v]
            
            # Check convergence
            diff = sum((v_new[i] - v[i]) ** 2 for i in range(n)) ** 0.5
            v = v_new
            if diff < tol:
                break
        
        return v

    def _deflate(self, A, eigenvector):
        """Deflate the matrix by removing the component along the eigenvector.
        
        Args:
            A (list of lists): Original matrix.
            eigenvector (list): Eigenvector to remove.
        
        Returns:
            list of lists: Deflated matrix.
        """
        n = len(A)
        # Compute eigenvalue (Rayleigh quotient)
        Av = [sum(A[i][j] * eigenvector[j] for j in range(n)) for i in range(n)]
        eigenvalue = sum(Av[i] * eigenvector[i] for i in range(n))
        
        # Outer product: eigenvector * eigenvector^T
        outer = [[eigenvector[i] * eigenvector[j] for j in range(n)] for i in range(n)]
        
        # Deflate: A - eigenvalue * (v * v^T)
        for i in range(n):
            for j in range(n):
                A[i][j] -= eigenvalue * outer[i][j]
        return A

    def fit(self, X):
        """Fit the PCA model to the data.
        
        Args:
            X (list of lists): Input data, where each inner list is a sample.
        """
        if not X or len(X) < 2:
            raise ValueError("X must have at least 2 samples")
        
        # Center the data
        X_centered = self._mean_center(X)
        
        # Compute covariance matrix
        cov_matrix = self._covariance_matrix(X_centered)
        
        # Extract top n_components eigenvectors using power iteration
        self.components = []
        temp_cov = [row[:] for row in cov_matrix]  # Copy to avoid modifying original
        for _ in range(min(self.n_components, len(X[0]))):
            eigenvector = self._power_iteration(temp_cov)
            self.components.append(eigenvector)
            temp_cov = self._deflate(temp_cov, eigenvector)

    def transform(self, X):
        """Transform the data into the principal component space.
        
        Args:
            X (list of lists): Input data to transform.
        
        Returns:
            list of lists: Transformed data.
        """
        if self.components is None:
            raise ValueError("Model must be trained with fit() before transforming")
        
        # Center the data using the stored mean
        X_centered = [[X[i][j] - self.mean[j] for j in range(len(X[0]))] for i in range(len(X))]
        
        # Project onto principal components
        components_T = self._transpose(self.components)
        return self._matrix_multiply(X_centered, components_T)


# Example usage
if __name__ == "__main__":
    # Synthetic data: 2D points with correlation
    X = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0]
    ]
    
    # Create and fit PCA model
    pca = PCA(n_components=1)
    pca.fit(X)
    
    # Transform data
    X_transformed = pca.transform(X)
    
    # Print results
    print("Principal components:", pca.components)
    print("Transformed data:", X_transformed)