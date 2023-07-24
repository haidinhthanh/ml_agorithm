import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt


class ImageUtils:

    def show_image(image):
        plt.imshow(image, cmap='gray')
        plt.axis('off')

    def show_images_comparision(original_X, reconstructed_X,
                                num_images, ini_figsize=[6,12]):
        num_cols = 2
        num_rows = num_images
        plt.figure(figsize=ini_figsize)
        for i in range(num_rows):
            plt.subplot(num_rows, num_cols, i*2+1)
            ImageUtils.show_image(original_X[i].reshape(64,64))
            plt.title(f"Original Image {i+1}")
            
            plt.subplot(num_rows, num_cols, (i+1)*2)
            ImageUtils.show_image(reconstructed_X[i].reshape(64,64))
            plt.title(f"Reconstructed Image {i+1}")
        plt.show()

    def plot_explained_variance(pca):
        plt.figure(figsize = (14, 8))
        nc = np.arange(1, pca.explained_variances.shape[0] + 1)
        plt.plot(nc, pca.explained_variances, 'b^')
        plt.plot(nc, pca.explained_variances, '--r')
        plt.xlabel('Number of PCA components', fontsize=16)
        plt.ylabel('Explained variance (%)', fontsize=16)
        plt.title('Explained variance by number of components', fontsize=16)
        plt.show()

    def plot_iris_in_3d(X_compressed, y):
        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
        ax.set_position([0, 0, 0.95, 1])
        for name, label in [("Setosa", 0),
                            ("Versicolour", 1),
                            ("Virginica", 2)]:
            ax.text3D(
                X_compressed[y == label, 0].mean(),
                X_compressed[y == label, 1].mean() + 1.5,
                X_compressed[y == label, 2].mean(),
                name,
                horizontalalignment="center",
                bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
            )

        # Reorder the labels to have colors matching the cluster results
        y = np.choose(y, [1, 2, 0]).astype(float)
        ax.scatter(X_compressed[:, 0],
                   X_compressed[:, 1],
                   X_compressed[:, 2],
                   c=y,
                   cmap=plt.cm.nipy_spectral,
                   edgecolor="k")

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        plt.show()
