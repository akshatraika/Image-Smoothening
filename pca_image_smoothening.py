# Author: Akshat Raika
# Project: P5 Eigenfaces
# Last Modified: 3/10/2020, 8:46PM
from scipy.io import loadmat
import numpy as np
from scipy.linalg import eigh
from matplotlib import pyplot
#  load the dataset from a provided .mat file, re-center it around the origin and return it as a NumPy array of floats
def load_and_center_dataset(filename):
    dataset = loadmat(filename)
    x = np.array(dataset['fea'])
    mean = np.mean(x, axis=0)
    x = x - mean
    return x

# calculate and return the covariance matrix of the dataset as a NumPy matrix (d x d array)
def get_covariance(dataset):
    temp = np.dot(np.transpose(dataset), dataset) / (len(dataset)-1)
    return temp

# perform eigen decomposition on the covariance matrix S and return a diagonal matrix (NumPy array)
# with the largest m eigenvalues on the diagonal, and a matrix (NumPy array) with the corresponding 
# eigenvectors as columns
# Return the eigenvalues as a diagonal matrix, in descending order, and the corresponding 
# eigenvectors as columns in a matrix.
def get_eig(S, m):
    w, v = eigh(S)
    w = w[-m:]
    w = np.flip(w, axis=0)
    v = np.transpose(v)
    v = v[-m:]
    v = np.flip(v, axis=0)
    v = v.T
    
    ev = np.identity(m, dtype=float)
    return np.multiply(w, ev), v
    
# project each image into your m-dimensional space and return the new representation as a d x 1 NumPy array
def project_image(image, U):
    alpha = np.dot(image, U)
    return np.dot(alpha, U.T)

# use matplotlib to display a visual representation of the original image and the projected image side-by-side
def display_image(orig, proj):
    orig = np.reshape(orig, (32,32)).T
    proj = np.reshape(proj, (32,32)).T
    fig, (ax1, ax2) = pyplot.subplots(figsize=(15, 5), ncols=2)
    ax1.set_title("Original")
    ax2.set_title("Projection")
    r0 = ax1.imshow(orig, aspect="equal")
    fig.colorbar(r0, ax=ax1)
    r1 = ax2.imshow(proj, aspect= "equal")
    fig.colorbar(r1, ax=ax2)
    fig.show()
    

