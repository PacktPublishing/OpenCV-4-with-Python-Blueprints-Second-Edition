import cv2
import numpy as np


UNIFORM_SIZE = (16, 16)


def resize_data(data, size=(32, 32)):
    return [cv2.resize(x, size) for x in data]



def hog_featurizer(data):
    """
    Featurize using histogram of gradients.

    For each image:
        1. resize all images to have the same (usually smaller size)
        2. Calculate HOG values using same HOGDescriptor for all images,
            so we get same number of gradients for each image
        3. Return a flattened list of gradients as a final feature.

    """
    uniform_size = (32, 32)

    block_size = (uniform_size[0] / 2, uniform_size[1] / 2)
    block_stride = (uniform_size[0] / 4, uniform_size[1] / 4)
    cell_size = block_stride
    num_bins = 9
    hog = cv2.HOGDescriptor()
    # hog = cv2.HOGDescriptor(small_size, block_size, block_stride,
    #                         cell_size, num_bins)

    return [hog.compute(x).flatten() for x in data]


def grayscale_featurize(data):
    """
    Featurize by calculating grayscale values of the data

    For each image:
        1. resize all images to have the same (usually smaller size)
        2. Convert the image to grayscale (values in 0 - 255 range)
        3. Convert each image to have pixel value in (0, 1) and flatten
        4. Subtract average pixel value of the flattened vector.
    """
    # uniform_size = (32, 32)

    resized_images = (cv2.resize(x, UNIFORM_SIZE) for x in data)
    gray_data = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in resized_images)
    scaled_data = (np.array(x).astype(np.float32).flatten() / 255
                   for x in gray_data)
    return np.vstack([x - x.mean() for x in scaled_data])


def load_data(rootpath="datasets/gtsrb_training", feature=None, cut_roi=True,
              test_split=0.2, plot_samples=False, seed=113):
    """Loads the GTSRB dataset

        This function loads the German Traffic Sign Recognition Benchmark
        (GTSRB), performs feature extraction, and partitions the data into
        mutually exclusive training and test sets.

        :param rootpath:     root directory of data files, should contain
                             subdirectories "00000" for all samples of class
                             0, "00004" for all samples of class 4, etc.
        :param feature:      which feature to extract: None, "gray", "rgb",
                             "hsv", surf", or "hog"
        :param cut_roi:      flag whether to remove regions surrounding the
                             actual traffic sign (True) or not (False)
        :param test_split:   fraction of samples to reserve for the test set
        :param plot_samples: flag whether to plot samples (True) or not
                             (False)
        :param seed:         which random seed to use
        :returns:            (X_train, y_train), (X_test, y_test)
    """
    # hardcode available class labels
    classes = np.arange(0, 42, 2)

    # read all training samples and corresponding class labels
    X = []  # data
    labels = []  # corresponding labels
    for c in range(len(classes)):
        # subdirectory for class
        prefix = rootpath + '/' + format(classes[c], '05d') + '/'

        # annotations file
        gt_file = open(prefix + 'GT-' + format(classes[c], '05d') + '.csv')

        # csv parser for annotations file
        gt_reader = csv.reader(gt_file, delimiter=';')
        next(gt_reader)  # skip header

        # loop over all images in current annotations file
        for row in gt_reader:
            # first column is filename
            im = cv2.imread(prefix + row[0])

            # remove regions surrounding the actual traffic sign
            if cut_roi:
                im = im[np.int(row[4]):np.int(row[6]),
                        np.int(row[3]):np.int(row[5]), :]

            X.append(im)
            labels.append(c)
        gt_file.close()

    # perform feature extraction
    X = _extract_feature(X, feature)

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    if plot_samples:
        num_samples = 15
        sample_idx = np.random.randint(len(X), size=num_samples)
        sp = 1
        for r in range(3):
            for c in range(5):
                ax = plt.subplot(3, 5, sp)
                sample = X[sample_idx[sp - 1]]
                ax.imshow(sample.reshape((32, 32)), cmap=cm.Greys_r)
                ax.axis('off')
                sp += 1
        plt.show()

    num_train = int(len(X) * (1 - test_split))

    X_train, y_train = X[:num_train], labels[:num_train]
    X_test, y_test = X[num_train:], labels[num_train:]

    return (X_train, y_train), (X_test, y_test)

