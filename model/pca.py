import numpy as np


def center_data(data):
    # Subtract the mean from each feature to center the data
    mean = np.mean(data, axis=0)
    return data - mean, mean


def calculate_covariance_matrix(centered_data):
    # Calculate the covariance matrix
    num_samples = centered_data.shape[0]
    covariance_matrix = np.dot(centered_data.T, centered_data) / num_samples
    return covariance_matrix


def calculate_eigenvectors(cov_matrix):
    # Calculate the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # Sort the eigenvectors in descending order based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvectors


def project_data(data, eigenvectors, num_dimensions):
    # Project the data onto the selected eigenvectors
    return np.dot(data, eigenvectors[:, :num_dimensions])


def main():
    # Sample data (replace this with your own dataset)
    data = np.array([[2, 3], [1, 2], [3, 4], [4, 3], [3, 6]])

    # Step 1: Center the data (subtract the mean)
    centered_data, mean = center_data(data)

    # Step 2: Calculate the covariance matrix
    cov_matrix = calculate_covariance_matrix(centered_data)

    # Step 3: Calculate the eigenvectors
    eigenvectors = calculate_eigenvectors(cov_matrix)

    # Step 4: Project the data onto the principal components (eigenvectors)
    num_dimensions = 1  # Number of dimensions to project onto (can be adjusted)
    projected_data = project_data(centered_data, eigenvectors, num_dimensions)

    # Optional: Reconstruct the data back from the projected data
    reconstructed_data = np.dot(projected_data, eigenvectors[:, :num_dimensions].T) + mean

    print("Original data:\n", data)
    print("Projected data:\n", projected_data)
    print("Reconstructed data:\n", reconstructed_data)


if __name__ == "__main__":
    main()
