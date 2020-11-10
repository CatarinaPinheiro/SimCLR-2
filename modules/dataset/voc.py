import os

import PIL

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']


def load_image(file):
    return PIL.Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


class VOC12(Dataset):

    def __init__(self, root, image_set, input_transform=None, target_transform=None):
        self.image_set = image_set
        if self.image_set in ["train", "trainval", "val"]:
            pass
        else:
            raise NameError

        split_main_dir = os.path.join(root, 'ImageSets/Segmentation')
        split_dir = os.path.join(split_main_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_dir), "r") as split:
            file_names = [x.strip() for x in split.readlines()]

        total_images_root = os.path.join(root, 'images')
        total_labels_root = os.path.join(root, 'labels')

        self.images_root = [os.path.join(total_images_root, x + ".jpg") for x in file_names]
        self.labels_root = [os.path.join(total_labels_root, x + ".png") for x in file_names]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        image = PIL.Image.open(self.images_root[index]).convert('RGB')
        label = PIL.Image.open(self.labels_root[index]).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.images_root)
