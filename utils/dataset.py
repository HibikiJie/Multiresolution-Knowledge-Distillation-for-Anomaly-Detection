import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy
import imgaug.augmenters as iaa


class ADDataset(Dataset):

    def __init__(self, root='data', mode='train'):
        super(ADDataset, self).__init__()
        self.dataset = []
        for image_name in os.listdir(f'{root}/{mode}'):
            image_path = f'{root}/{mode}/{image_name}'
            self.dataset.append(image_path)
        self.seq = iaa.SomeOf((0, None), [
            iaa.Fliplr(1.0),
            iaa.Flipud(1.0),
            # iaa.GaussianBlur(1.0)
        ], random_order=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image_path = self.dataset[item]
        image = cv2.imread(image_path)
        image = self.square_picture(image, 224)
        image = self.seq.augment_images([image])[0]
        # cv2.imshow('a', image)
        # cv2.waitKey()

        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
        return image_tensor

    @staticmethod
    def square_picture(image, image_size):
        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        if max_len >= image_size:
            fx = image_size / max_len
            fy = image_size / max_len
            image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
            h2, w2, _ = image.shape
            background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
            background[:, :, :] = 0
            s_h = image_size // 2 - h2 // 2
            s_w = image_size // 2 - w2 // 2
            background[s_h:s_h + h2, s_w:s_w + w2] = image
            return background
        else:
            h2, w2, _ = image.shape
            background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
            background[:, :, :] = 0
            s_h = image_size // 2 - h2 // 2
            s_w = image_size // 2 - w2 // 2
            background[s_h:s_h + h2, s_w:s_w + w2] = image
            return background


if __name__ == '__main__':
    data = ADDataset('/media/cq/data/public/hibiki/Knowledge_Distillation/data')
    for i in range(len(data)):
        data[i]
