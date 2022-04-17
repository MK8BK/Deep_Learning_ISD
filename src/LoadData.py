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

def flatten_img(nimg):
    """
        Returns the flattened representation of an image given as np.array
        @param: nimg : np.array of shape(rows, columns, 1), the image, px 0-255
        @return: flattened version: a normalized (0-1) column vector  (h*w, 1)
    """
    h, w = nimg.shape[0], nimg.shape[1]
    return nimg.flatten(order="F").reshape((h*w,1))/255.


def load_numpy_image(str_path: str) -> np.array:
    """
        Returns the flattened numpy representation of an image 
                            given the full relative str_path
        @param: str_path: the full relative str_path to the image file
        @return: im: a normalized np.array of dimensions (h*w,1)
    """
    pil_img = load_pil_image(str_path)
    numpy_image = np.array(pil_img)
    return flatten_img(numpy_image)


def make_input_matrix(samples: list[np.array]) -> np.array:
    """
        Returns the matrix representation of s samples
        @param: samples: a list of flattened np.array 's, 
                        each representing a sample image
        @return: input_matrix: a matrix (2d np.array) containing one sample per column,
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


def make_labels_matrix(labels: list[str], classes: list[int]=CLASSES) -> np.array:
    """
        Returns the one hot encoded matrix representation of the labels, 
                            given a list of char labels
        @param: labels: a list of strings, filepaths
        @return: labels_matrix: a 2 np.array of 16 rows, each column is an image
    """
    labels_matrix = np.zeros((len(classes), len(labels)))
    labels = make_labels(labels)
    for label, i in zip(labels, range(len(labels))):
        labels_matrix[label, i] = 1
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

def load_training_set(path_str: str, batch_size: int, classes: list[str]=CLASSES,
                            equilibrium: bool=True) -> tuple[np.array]:
    """
        Returns a training input and labels matrices randomly, 
                                        equal per class or not
        @param: path: the path to the data_set folder
        @param: batch_size: the number of images in the batch
        @param: classes: a list of str representations of the classes
        @param: equilibrium: a bool, wether or not to equalize images per class
        @return: batch: X: an input matrix of shape (784,batch_size) 
                        Y: a corresponding labels matrix of shape (16, batch_size)
    """
    files = make_random_batch(path_str, batch_size, classes, equilibrium)
    samples = [load_numpy_image(file) for file in files]
    X = make_input_matrix(samples)
    Y = make_labels_matrix(files, classes)
    return X,Y

def load_prediction_image(path_str: str):
    """
        Loads a single image located at path_str
        @param: path_str: relative path to the image
        @return: x: a 784x1 input matrix
                 im: PIL representation of the image
    """
    im = load_pil_image(path_str)
    nim = load_numpy_image(path_str).reshape((im.width*im.height, 1))
    x = make_input_matrix([nim])
    #y = make_labels_matrix([path_str])
    return (x, im)#(x, nim, im, y)



def load_data_set(path_str):
    """
        Loads entire data set at path_str
        @param: path_str: root path of data_set
        @return: X: an imput matrix of shape (784, 38400)
                 Y: a labels matrix of shape (16, 38400)
    """
    directories = listdir(path_str)
    imgs_per_directory = [listdir(path_str+directory) for directory in directories]
    imgs = [path_str+img[0]+'/'+img for directory in imgs_per_directory 
            for img in directory]
    samples = [load_numpy_image(file) for file in imgs]
    X = make_input_matrix(samples)
    Y = make_labels_matrix(imgs)
    return X, Y




if __name__ == "__main__":
    #help(make_labels)
    X, Y = load_data_set("EMNIST_DATA_SET/")
    print(X.shape, Y.shape)
    #print(Y)
    #print(X, "\n\n", Y, "\n")
    #X_test, Y_test, X_train, Y_train = split_data(X, Y, percent_test=25)
    #print(X_test.shape, "\n", Y_test.shape, "\n", X_train.shape, "\n", Y_train.shape)
    #print("No errors")
