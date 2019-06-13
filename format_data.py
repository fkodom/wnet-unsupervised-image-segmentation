import os
from datetime import datetime

import h5py
import numpy as np
from PIL import Image


def get_matching_file_names(root: str):
    labels = os.listdir(os.path.join(root, 'train_masks'))
    file_names = []

    for label in labels:
        split_label = label[:-4].split('_')
        split_label.remove('mask')
        image = '_'.join(split_label) + '.jpg'

        image_path = os.path.join(root, 'train', image)
        label_path = os.path.join(root, 'train_masks', label)
        file_names.append((label_path, image_path))

    return file_names


def get_mask(path, shape=None):
    img = Image.open(path)
    if shape:
        img = img.resize(shape, Image.BILINEAR)

    return np.array(img).astype(np.uint8)


def get_raw_image(path, shape=None):
    img = Image.open(path)
    if shape:
        img = img.resize(shape, Image.BILINEAR)

    img = np.array(img)
    img = np.moveaxis(img, -1, 0)

    return img.astype(np.uint8)


def get_data_set(root: str, shape=None):
    file_names = get_matching_file_names(root)
    labels = np.array([get_mask(label_path, shape) for label_path, _ in file_names])
    images = np.array([get_raw_image(image_path, shape) for _, image_path in file_names])

    return labels, images


if __name__ == '__main__':
    train_dir = 'data'
    validation_split = 0.3
    image_shape = (192, 128)
    output_file = os.path.join('data', 'train-small.hdf5')

    print('Getting data...')
    train_labels, train_images = get_data_set(train_dir, image_shape)
    print(train_images.shape, train_labels.shape)

    print('Generating training & validation sets...')
    N = int(train_labels.shape[0] * validation_split)
    val_labels, val_images = train_labels[:N], train_images[:N]
    train_labels, train_images = train_labels[N:], train_images[N:]

    print('Saving to file...')
    with h5py.File(output_file, 'w') as f:
        f.attrs['Date Created'] = str(datetime.now())
        f.attrs['Data Format'] = '(batch_size, channel, row, column)'
        f.attrs['Training Examples'] = len(train_images)
        f.attrs['Validation Examples'] = len(val_images)

        f.create_group('Training')
        f['Training'].create_dataset('Inputs', data=train_images, compression='gzip')

        f.create_group('Validation')
        f['Validation'].create_dataset('Inputs', data=val_images, compression='gzip')
