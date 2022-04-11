from os import listdir, path
from random import choice as listChoice
from random import shuffle as listShuffle
import numpy as np
from PIL import Image, ImageOps


#global classes list
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
            , "A", "B", "C", "D", "E", "F"]


def char_to_label(character: str) -> int:
    """
        Convert the character representation of the class into its integer representation
        @param: character: str , has to be in the global defined classes
        @return: the integer representation of said character
    """
    assert(character in CLASSES), \
        f"'{character}' not in classes = {CLASSES}"
    return CLASSES.index(character)


def label_to_char(label: int) -> str:
    """
        Convert the integer representation of the class into its character representation
        @param: label: int , has to be a valid index of the global defined classes
        @return: the character representation of said integer
    """
    assert(label<len(CLASSES)),\
        f"""no character encoded by {label} in classes = {CLASSES} of length {len(CLASSES)}"""
    return CLASSES[label]


def load_pil_image(str_path: str) -> Image:
    """
        Returns a grayscale PIL image given the full relative str_path
        @param: str_path: the full relative str_path to the image file
        @return: im: a PIL Image object (single channel: grayscale)
    """
    temp = Image.open(str_path)
    if "F" in str_path:
        im = temp.rotate(180)
    else:
        im = temp.rotate(270)
    temp.close()
    im = ImageOps.grayscale(im)
    return im


def load_numpy_image(str_path: str) -> np.array:
    """
        Returns the flattened numpy representation of an image 
                            given the full relative str_path
        @param: str_path: the full relative str_path to the image file
        @return: im: a normalized np.array of dimensions (1,h*w)
    """
    pil_img = load_pil_image(str_path)
    h,w = pil_img.size
    return np.array(pil_img).flatten(order="F")/255.


def make_input_matrix(samples: list[np.array]) -> np.array:
    """
        Returns the matrix representation of s samples
        @param: samples: a list of flattened np.array 's, 
                        each representing a sample image
        @return: input_matrix: a matrix containing one sample per column,
                                         1 feature(pixel value) per row
    """
    sample_shape = samples[0].shape
    for sample in samples:
        assert(sample_shape==sample.shape), \
            f"""samples of different shape: sample 0       : {sample_shape}
                                            other sample   : {sample.shape}"""
    input_matrix = np.concatenate(samples).reshape((-1, len(samples)), order="F")
    return input_matrix


def make_labels(filenames: list[str]) -> np.array:
    """
        Returns the class label for each image filename in filenames
        @param: filenames: a list of strings, relative str_paths to files
        @return: labels: a list of int labels : the class of each file
    """
    labels = [char_to_label(filename[18:19]) for filename in filenames]
    return labels


def make_labels_matrix(labels: list[str], classes: list[int]) -> np.array:
    """
        Returns the matrix representation of the labels, 
                            given a list of char labels
        @param: labels: a list of strings, all part of global CLASSES
        @return: classes: the global CLASSES list
    """
    #print(len(labels))
    labels_matrix = np.zeros((len(classes), len(labels)))
    labels = make_labels(labels)
    for label, i in zip(labels, range(len(labels))):
        labels_matrix[label, i] = 1
    #labels_matrix.dtype = int
    return labels_matrix.astype("int32")


def make_random_batch(path: str, batch_size: int, classes: list[str],
                        equilibrium: bool=True)->list[str]:
    """
        Returns a list of filenames randomly, equal per class or not
        @param: path: the path to the data_set folder
        @param: batch_size: the number of images in the batch
        @param: classes: a list of str representations of the classes
        @param: equilibrium: a bool, wether or not to equalize images per class
        @return: batch: a list of strings, 
                the paths to the randomly selected images
    """
    #getting directories
    directories = listdir(path)
    #verifying the correct number of classes
    assert(len(directories)==len(classes)), \
        f"directories {len(directories)} do not match classes {len(classes)}"
    #list of lists: images per subdirectory of the data folder
    imgs_per_directory = [listdir(path+directory) for directory in directories]
    #case where equal number of images per class
    if equilibrium:
        #verifying that the batch can be divided equally
        assert(batch_size%len(classes)==0), \
            f"can't divide batch size {batch_size} equally among {len(classes)} classes"
        
        perClass = batch_size//len(classes)
        batch = [listChoice(directory) for directory in imgs_per_directory 
                    for i in range(perClass)]
        batch = [path+img[0]+'/'+img for img in batch]
        listShuffle(batch)
        return batch
    
    else:
        imgs = [path+img[0]+'/'+img for directory in imgs_per_directory 
                    for img in directory]
        batch = [listChoice(imgs) for i in range(batch_size)]
        return batch

def load_data_set(path_str: str, batch_size: int, classes: list[str],
                            equilibrium: bool=True) -> list[np.array]:
    files = make_random_batch(path_str, batch_size, classes, equilibrium)
    samples = [load_numpy_image(file) for file in files]
    X = make_input_matrix(samples)
    Y = make_labels_matrix(files, classes)
    return X,Y

def load_image(path_str, classes):
    im = load_pil_image(path_str)
    nim = load_numpy_image(path_str).reshape((im.width*im.height, 1))
    x = make_input_matrix([nim])
    y = make_labels_matrix([path_str], classes)
    return (x, y, nim, im)
#Draft of normalization, might be useful
#X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train)
#X_train, X_test = X_train / np.std(X_train), X_test / np.std(X_train)

def split_data(X: np.array, Y: np.array, 
                percent_test: int=30) -> tuple[np.array, np.array,
                                                np.array, np.array]:
    
    #assert same number of samples and labels
    assert(X.shape[1]==Y.shape[1]), \
        f"Y labels of {Y.shape[1]} samples dont match X data of {X.shape[1]} samples"

    m = X.shape[1]
    test_samples = int((m*percent_test)/100)
    X_test = np.zeros((X.shape[0], test_samples))
    X_train = X
    Y_test = np.zeros((Y.shape[0], test_samples))
    Y_train = Y
    for i in range(test_samples):
        to_be_removed = listChoice(list(range(X_train.shape[1])))
        X_test[:, i] = X_train[:, to_be_removed]
        X_train = np.delete(X_train, i, 1)
        Y_test[:, i] = Y_train[:, to_be_removed]
        Y_train = np.delete(Y_train, i, 1)

    return X_test, Y_test, X_train, Y_train

if __name__ == "__main__":
    #help(make_labels)
    X, Y = load_data_set("EMNIST_DATA_SET/", batch_size=3,
                            classes=CLASSES, equilibrium=False)
    #print(Y)
    #print(X, "\n\n", Y, "\n")
    #X_test, Y_test, X_train, Y_train = split_data(X, Y, percent_test=25)
    #print(X_test.shape, "\n", Y_test.shape, "\n", X_train.shape, "\n", Y_train.shape)
    print("No errors")
