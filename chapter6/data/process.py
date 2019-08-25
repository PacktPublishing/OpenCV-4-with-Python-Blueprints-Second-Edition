import cv2
import numpy as np


# UNIFORM_SIZE = (16, 16)
UNIFORM_SIZE = (32, 32)


def hog_featurize(data):
    """
    Featurize using histogram of gradients.

    For each image:
        1. resize all images to have the same (usually smaller size)
        2. Calculate HOG values using same HOGDescriptor for all images,
            so we get same number of gradients for each image
        3. Return a flattened list of gradients as a final feature.

    """
    block_size = (UNIFORM_SIZE[0] // 2, UNIFORM_SIZE[1] // 2)
    block_stride = (UNIFORM_SIZE[0] // 4, UNIFORM_SIZE[1] // 4)
    cell_size = block_stride
    hog = cv2.HOGDescriptor(UNIFORM_SIZE, block_size, block_stride,
                            cell_size, 9)

    resized_images = (cv2.resize(x, UNIFORM_SIZE) for x in data)

    return np.array([hog.compute(x).flatten() for x in resized_images])


def grayscale_featurize(data):
    """
    Featurize by calculating grayscale values of the data

    For each image:
        1. resize all images to have the same (usually smaller size)
        2. Convert the image to grayscale (values in 0 - 255 range)
        3. Convert each image to have pixel value in (0, 1) and flatten
        4. Subtract average pixel value of the flattened vector.
    """
    resized_images = (cv2.resize(x, UNIFORM_SIZE) for x in data)
    gray_data = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in resized_images)
    scaled_data = (np.array(x).astype(np.float32).flatten() / 255
                   for x in gray_data)
    return np.vstack([x - x.mean() for x in scaled_data])


if __name__ == '__main__':
    from data.gtsrb import load_training_data
    import matplotlib.pyplot as plt

    train_data, train_labels = load_training_data(labels=[13])

    i = 80

    [f] = hog_featurize([train_data[i]])
    print(len(f))

    plt.imshow(train_data[i])
    plt.show()
