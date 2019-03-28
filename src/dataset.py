import os
import torch
#from skimage import io, transform
from PIL import Image
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# class Rescale(object):
#     """Rescale the image in a sample to a given size.
#     Args:
#         output_size (tuple): Desired output size.
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, tuple)
#         assert len(output_size) == 2
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, labels = sample['image'], sample['labels']
#         img = transform.resize(image, self.output_size)
#         return {'image': img, 'labels': labels}


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         image, labels = sample['image'], sample['labels']

#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'labels': torch.from_numpy(labels)}


# REF: https://pytorch.org/docs/stable/torchvision/models.html
# Note: PIL.Image.open(filename) returns a PIL image
#       skimage.io.imread(filename) returns a numpy ndarray
image_size = (224, 224)
# image_size = (299, 299) # torchvision.models.resnet152 will not work properly for this size (i.e., size mismatch error)
data_transforms = transforms.Compose([transforms.Resize(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])])


class ImageDataset(Dataset):
    """Image dataset."""

    def __init__(self, image_dir, image_list_file, label_file, vocab_file, transform=data_transforms):
        """
        Args:
            image_dir (string): Directory with all the images.
            image_list_file (string): Path to the list of image filenames.
            label_file (string): Path to the .mat file of image labels.
            vocab_file (string): Path to the vocabulary file of labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        image_files = []
        with open(image_list_file, 'r') as fd:
            for line in fd:
                image_files.append(line.strip())
        
        labels_dict = loadmat(label_file)
        if 'mat_train' in labels_dict:
            all_labels = labels_dict['mat_train'].astype(np.uint8)
        else:
            all_labels = labels_dict['mat_test'].astype(np.uint8)

        # filtering out examples without any label
        nlabels = all_labels.sum(axis=1)
        if np.min(nlabels) < 1:
            nz_ix = np.nonzero(nlabels)[0]
            self.image_files = np.array(image_files)[nz_ix]
            self.all_labels = all_labels[nz_ix, :]
        else:
            self.image_files = image_files
            self.all_labels = all_labels
        
        vocab = []
        with open(vocab_file, 'r') as fd:
            for line in fd:
                vocab.append(line.strip())
        self.label_vocab = np.array(vocab)
        
        self.num_samples, self.num_labels = self.all_labels.shape
        assert self.num_samples == len(self.image_files)
        assert self.num_labels == len(self.label_vocab)
        
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        #image = io.imread(img_name)
        image = Image.open(img_name)
        img_labels = self.all_labels[idx, :]
        
        # convert grayscale image to rgb color image
        rgbmode = 'RGB'
        if image.mode != rgbmode:
            image = image.convert(mode=rgbmode)
        
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'labels': img_labels}

        return sample
    
    def decode_labels(self, label_vec):
        if type(label_vec) == torch.Tensor:
            label_vec = label_vec.numpy()
        assert label_vec.shape == (self.num_labels,)
        label_ix = np.nonzero(label_vec)[0]
        return self.label_vocab[label_ix].tolist()
