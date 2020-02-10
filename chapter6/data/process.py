import cv2
import numpy as np
import itertools


def hog_featurize(data, *, scale_size=(32, 32)):
    """
    Featurize using histogram of gradients.

    For each image:
        1. resize all images to have the same (usually smaller size)
        2. Calculate HOG values using same HOGDescriptor for all images,
            so we get same number of gradients for each image
        3. Return a flattened list of gradients as a final feature.

    """
    block_size = (scale_size[0] // 2, scale_size[1] // 2)
    block_stride = (scale_size[0] // 4, scale_size[1] // 4)
    cell_size = block_stride
    hog = cv2.HOGDescriptor(scale_size, block_size, block_stride,
                            cell_size, 9)

    resized_images = (cv2.resize(x, scale_size) for x in data)

    return np.array([hog.compute(x).flatten() for x in resized_images])


def surf_featurize(data, *, scale_size=(16, 16), num_surf_features=100):
    all_kp = [cv2.KeyPoint(float(x), float(y), 1)
              for x, y in itertools.product(range(scale_size[0]),
                                            range(scale_size[1]))]
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=400)
    kp_des = (surf.compute(x, all_kp) for x in data)
    return np.array([d.flatten()[:num_surf_features]
                     for _, d in kp_des]).astype(np.float32)


def hsv_featurize(data, *, scale_size=(16, 16)):
    """
    Featurize by calculating HSV values of the data

    For each image:
        1. resize all images to have the same (usually smaller size)
        2. Convert the image to HSV (values in 0 - 255 range)
        3. Convert each image to have pixel value in (0, 1) and flatten
        4. Subtract average pixel value of the flattened vector.
    """
    resized_images = (cv2.resize(x, scale_size) for x in data)
    hsv_data = (cv2.cvtColor(x, cv2.COLOR_BGR2HSV) for x in resized_images)
    scaled_data = (np.array(x).astype(np.float32).flatten() / 255
                   for x in hsv_data)
    return np.vstack([x - x.mean() for x in scaled_data])


def grayscale_featurize(data, *, scale_size=(16, 16)):
    """
    Featurize by calculating grayscale values of the data

    For each image:
        1. resize all images to have the same (usually smaller size)
        2. Convert the image to grayscale (values in 0 - 255 range)
        3. Convert each image to have pixel value in (0, 1) and flatten
        4. Subtract average pixel value of the flattened vector.
    """
    resized_images = (cv2.resize(x, scale_size) for x in data)
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
