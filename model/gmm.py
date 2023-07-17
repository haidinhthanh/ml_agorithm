import numpy as np


class GaussianMixtureModel:
    def __init__(self, n_components, n_features, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.n_features = n_features
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        self._initialize_parameters(X)

        for _ in range(self.max_iter):
            prev_likelihood = self._compute_log_likelihood(X)
            self._expectation_step(X)
            self._maximization_step(X)
            curr_likelihood = self._compute_log_likelihood(X)

            if abs(curr_likelihood - prev_likelihood) < self.tol:
                break

    def _initialize_parameters(self, X):
        self.pi = np.full(shape=self.n_components, fill_value=1 / self.n_components)
        self.mu = np.random.rand(self.n_components, self.n_features)
        self.sigma = np.array([np.eye(self.n_features)] * self.n_components)

    def _compute_log_likelihood(self, X):
        likelihood = np.zeros((len(X), self.n_components))

        for k in range(self.n_components):
            diff = X - self.mu[k]
            exponent = -0.5 * np.sum(np.dot(diff, np.linalg.inv(self.sigma[k])) * diff, axis=1)
            coefficient = 1 / np.sqrt((2 * np.pi) ** self.n_features * np.linalg.det(self.sigma[k]))
            likelihood[:, k] = coefficient * np.exp(exponent)

        return np.log(np.sum(likelihood, axis=1)).sum()

    def _expectation_step(self, X):
        self.gamma = np.zeros((len(X), self.n_components))

        for k in range(self.n_components):
            diff = X - self.mu[k]
            exponent = -0.5 * np.sum(np.dot(diff, np.linalg.inv(self.sigma[k])) * diff, axis=1)
            coefficient = 1 / np.sqrt((2 * np.pi) ** self.n_features * np.linalg.det(self.sigma[k]))
            self.gamma[:, k] = self.pi[k] * coefficient * np.exp(exponent)

        self.gamma = self.gamma / np.sum(self.gamma, axis=1, keepdims=True)

    def _maximization_step(self, X):
        for k in range(self.n_components):
            Nk = np.sum(self.gamma[:, k])

            self.pi[k] = Nk / len(X)
            self.mu[k] = np.sum(self.gamma[:, k].reshape(-1, 1) * X, axis=0) / Nk
            diff = X - self.mu[k]
            self.sigma[k] = np.dot((diff * self.gamma[:, k].reshape(-1, 1)).T, diff) / Nk

    def predict(self, X):
        likelihood = np.zeros((len(X), self.n_components))

        for k in range(self.n_components):
            diff = X - self.mu[k]
            exponent = -0.5 * np.sum(np.dot(diff, np.linalg.inv(self.sigma[k])) * diff, axis=1)
            coefficient = 1 / np.sqrt((2 * np.pi) ** self.n_features * np.linalg.det(self.sigma[k]))
            likelihood[:, k] = coefficient * np.exp(exponent)

        return np.argmax(likelihood, axis=1)


# Generate some sample data
np.random.seed(42)
n_samples = 1000
n_features = 2
n_components = 4

# Generate random data from three Gaussian distributions
means = np.array([[0, 0], [3, 3], [-3, 3]])
covariances = np.array([[[1, 0], [0, 1]], [[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]]])
X = np.concatenate([np.random.multivariate_normal(mean, cov, int(n_samples / n_components))
                    for mean, cov in zip(means, covariances)])

# Create and fit the Gaussian Mixture Model
gmm = GaussianMixtureModel(n_components=n_components, n_features=n_features)
gmm.fit(X)

# Make predictions on new data
new_data = np.array([[1, 1], [-2, 4], [0, 0]])
predictions = gmm.predict(X)
print("Predictions:", predictions)
