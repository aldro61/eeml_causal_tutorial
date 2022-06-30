
import base64
import numpy as np
import pandas as pd

from IPython.display import HTML
from io import BytesIO
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision import transforms


def pprint_ates(ate):
    """
    Pretty print average treatment effect table
    
    """
    return HTML(pd.DataFrame([dict(method=k, ATE=v) for k, v in ate.items()]).to_html(index=False))


def generate_data_ex1(n_samples=10000, seed=42):
    random = np.random.RandomState(seed)

    Z = random.randint(0, 3, size=n_samples)
    
    policy = [0.05, 0.5, 0.9]  # Probability of treatment for each Z
    A = np.array([random.rand() <= policy[z] for z in Z]).astype(int)

    # Value of Z1: 0     1    2
    survival = [[0.75, 0.25, 0.1],  # Untreated (A = 0)
                [0.95, 0.75, 0.25]]  # Treated (A = 1)
    Y = np.fromiter((random.rand() <= survival[A[i]][Z[i]] for i in range(n_samples)), dtype=int)

    return pd.DataFrame(dict(Z=Z, A=A, Y=Y))


def image_base64(img_array):
    """
    Convert an image from numpy array to bytes
    
    Adapted from https://www.kaggle.com/code/stassl/displaying-inline-images-in-pandas-dataframe/notebook

    """
    with BytesIO() as buffer:
        im = Image.fromarray(img_array)
        im.thumbnail((200, 200), Image.LANCZOS)
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    """
    Format image to show inline in dataframe

    Adapted from https://www.kaggle.com/code/stassl/displaying-inline-images-in-pandas-dataframe/notebook

    """
    return f'<img src="data:image/jpeg;base64,{im}" width="100">'


def generate_data_ex4b(randomize=False, seed=42):
    """
    Generate a dataset where confounding arises from a high-dimensional visual signal.

    We use images of cats and dogs from CIFAR10 to serve as confounder.

    """
    random = np.random.RandomState(seed)

    # Load image dataset
    base_dataset = CIFAR10(".", download=True)

    # Normalization applied to images
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Extract the list of indices for all individuals with class cat (3) and dog (5)
    class1_idx = np.where(np.array(base_dataset.targets) == 3)[0].tolist()
    class2_idx = np.where(np.array(base_dataset.targets) == 5)[0].tolist()

    # Keep only a few images and make them occur more than once
    # By doing this, we aim to create profiles of individuals
    # with the same visual profile (related to stratification)
    class1_idx = class1_idx[: 100] * 100
    class2_idx = class2_idx[: 100] * 100
    n_samples = len(class1_idx) + len(class2_idx)

    # Retrieve images from dataset and apply transformations
    Z = np.array(base_dataset.data[class1_idx + class2_idx])
    Z_normed = np.array([transform(z).numpy() for z in Z])

    # We assume that treatment was assigned based on perfect knowledge of the classes.
    #       This would avoid creating a parallel with latent variables (ask Dhanya).
    Z_ = np.array([0] * len(class1_idx) + [1] * len(class2_idx))

    if randomize:
        policy = [0.5, 0.5]
    else:
        policy = [0.3, 0.7]  # Probability of treatment for each Z
    A = np.array([random.rand() <= policy[z] for z in Z_]).astype(int)

    # Value of Z:  0     1
    survival = [[0.75, 0.10],  # Untreated (A = 0)
                [0.95, 0.25]]  # Treated (A = 1)    
    Y = np.fromiter((random.rand() <= survival[A[i]][Z_[i]] for i in range(n_samples)), dtype=int)

    # Create a random permutation for the examples
    shuffler = np.arange(n_samples)
    random.shuffle(shuffler)

    # Assemble final dataset
    return pd.DataFrame(dict(Z=[z for z in Z_normed[shuffler]], A=A[shuffler], Y=Y[shuffler], 
                             Z_img=[image_base64(z) for z in Z[shuffler]], Z_=Z_[shuffler]))
