import os
from utils.loss_functions import LossFunction
from models.network import ClonerNetwork, SourceNetwork
import matplotlib.pyplot as plt
from torch import nn
from scipy.ndimage.filters import gaussian_filter
import torch
import cv2
import numpy


class Detector:

    def __init__(self,is_cuda=False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and is_cuda else 'cpu')
        self.cloner_network = ClonerNetwork().to(self.device)
        self.source_network = SourceNetwork().to(self.device)
        self.cloner_network.load_state_dict(torch.load('weight/cloner_network.pt',map_location='cpu'))
        self.cloner_network.eval()
        self.source_network.eval()
        # self.loss_function = LossFunction(1)

    def detect(self, image):
        image = self.square_picture(image, 224)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)/255
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_(True)
        # print(image_tensor)
        self.cloner_network.zero_grad()
        predict = self.cloner_network(image_tensor)
        real = self.source_network(image_tensor)

        '''count loss'''
        y_pred_1, y_pred_2, y_pred_3 = predict[1:]
        y_1, y_2, y_3 = real[1:]
        criterion = nn.MSELoss()
        similarity_loss = torch.nn.CosineSimilarity()
        abs_loss_1 = criterion(y_pred_1, y_1)
        loss_1 = torch.mean(1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
        abs_loss_2 = criterion(y_pred_2, y_2)
        loss_2 = torch.mean(1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
        abs_loss_3 = criterion(y_pred_3, y_3)
        loss_3 = torch.mean(1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))
        total_loss = loss_1 + loss_2 + loss_3 + 0.5 * (abs_loss_1 + abs_loss_2 + abs_loss_3)

        total_loss.backward()
        # grad = torch.relu(image_tensor.grad.data.detach()).cpu().squeeze(0).numpy()
        grad = image_tensor.grad.data.detach().cpu().squeeze(0).numpy()

        # print(grad.max(),grad.min())
        grad = self.convert_to_grayscale(grad)
        # gbp_saliency = abs(grad)
        gbp_saliency = abs(grad)
        gbp_saliency = (gbp_saliency - min(gbp_saliency.flatten())) / (
                max(gbp_saliency.flatten()) - min(gbp_saliency.flatten()))
        # print(gbp_saliency.max(), gbp_saliency.min())
        saliency = gbp_saliency
        saliency = gaussian_filter(saliency, sigma=4)
        # print(saliency.max(), saliency.min())
        # print(saliency.shape)
        # cam = numpy.transpose(saliency,axes=(1,2,0))
        cam = saliency[0]
        # cam = cam/cam.max()
        # image = (image-image.min())/(image.max()-image.min())*255
        cam_map = numpy.uint8(255 * cam)
        contidion = cam < 0.5
        # cam_map[contidion] = 0
        # heatmap = cam_map[~contidion]
        heatmap = cv2.applyColorMap(cam_map, cv2.COLORMAP_JET)
        heatmap[contidion] = 0
        superimposed_img = cv2.addWeighted(image, 0.8, heatmap, 0.4, 0)

        # image = image.astype(numpy.uint8)
        cv2.imshow('a',superimposed_img)
        cv2.imshow('b', heatmap)
        cv2.imshow('c', image)
        cv2.waitKey()
        pass

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
    @staticmethod
    def convert_to_grayscale(im_as_arr):
        grayscale_im = numpy.sum(numpy.abs(im_as_arr), axis=0)
        im_max = numpy.percentile(grayscale_im, 99)
        im_min = numpy.min(grayscale_im)
        grayscale_im = (numpy.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
        grayscale_im = numpy.expand_dims(grayscale_im, axis=0)
        return grayscale_im

if __name__ == '__main__':
    detector = Detector()
    root = 'data/leather/test/fold'
    # root = 'data/leather/train'

    for image_name in os.listdir(root):
        image_path = f'{root}/{image_name}'
        image = cv2.imread(image_path)
        detector.detect(image)
        # exit()

