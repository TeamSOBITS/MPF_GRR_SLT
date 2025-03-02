import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import torch.nn.functional as F

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

# from .model import Net
from .osnetmodel import osnet_x1_0
import torch.nn as nn

class Extractor(object):
    def __init__(self, use_reid=True, use_cuda=True):
        self.net = osnet_x1_0(pretrained=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.net.to(self.device)

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Initialize the ROS publisher and CvBridge
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/mono_following/feature_maps', Image, queue_size=10)  # Adjust topic name as needed

    def _preprocess(self, im_crops):
        """
        Normalize and preprocess image crops for model input.
        """
        """
        TODO:
            1. to float with scale from 0 to 1
            2. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        
        im_batch = torch.cat(
            [self.norm(im.astype(np.float32) / 255.).unsqueeze(0) for im in im_crops], dim=0
        ).float()
        return im_batch

    def _normalize_feature_map(self, feature_maps):
        """
        Normalizes a batch of feature maps using L2 normalization.
        """
        # feature_maps = feature_maps ** 2
        feature_maps = feature_maps.sum(1)  # Sum over channel dimension: (b, c, h, w) -> (b, h, w)
        b, h, w = feature_maps.size()
        feature_maps = feature_maps.view(b, h * w)
        feature_maps = F.normalize(feature_maps, p=2, dim=1)  # L2 normalization
        feature_maps = feature_maps.view(b, h, w)
        return feature_maps

    def visualize_activation_map(self, img, feature_map):
        """
        Visualizes and optionally overlays the activation map on the input image.
        """
        # # Convert RGB to BGR if needed (OpenCV uses BGR)
        if img.shape[-1] == 3:  # Check if the image has 3 channels (RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Normalize feature map for visualization
        feature_map = feature_map.cpu().numpy()
        feature_map = feature_map.sum(axis=0)  # (h, w)
        feature_map = cv2.resize(feature_map, (img.shape[1], img.shape[0]))
        feature_map = 255 * (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-12)
        feature_map = np.uint8(feature_map)
        feature_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)

        # Overlay activation map on the original image
        overlayed = img * 0.3 + feature_map * 0.7
        overlayed = overlayed.clip(0, 255).astype(np.uint8)
        return overlayed

    def __call__(self, im_crops, visualize=True):
        """
        Runs the model and optionally visualizes the feature maps.
        """
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            self.net.eval()
            im_batch = im_batch.to(self.device)

            # Get the feature vectors and feature maps from the model
            feature_vectors = self.net(im_batch, return_features=True)  # 特徴ベクトルを取得
            feature_maps = self.net(im_batch, return_featuremaps=True)  # 特徴マップを取得

            # # Normalize the feature maps (if needed)
            # feature_maps = self._normalize_feature_map(feature_maps)

            if visualize:
                # Create an empty list to hold the overlayed images
                overlayed_images = []
                for i, feature_map in enumerate(feature_maps):
                    img = im_crops[i]
                    overlayed = self.visualize_activation_map(img, feature_map)
                    overlayed_images.append(overlayed)

                # Concatenate images horizontally (you can change this to vertical concatenation if preferred)
                concatenated_img = np.concatenate(overlayed_images, axis=1)  # Horizontal concatenation

                # Convert the concatenated image to a ROS message and publish it
                ros_img = self.bridge.cv2_to_imgmsg(concatenated_img, encoding="bgr8")
                self.pub.publish(ros_img)

        return feature_vectors.cpu().numpy()


class PersonExtractor(Extractor):
    def __init__(self, use_reid=True, use_cuda=True):
        Extractor.__init__(self, use_reid=use_reid, use_cuda=use_cuda)
    
    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _get_feature(self, im_crops, visualize=True):
        """
        Runs the model and optionally visualizes the feature maps.
        """
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            self.net.eval()
            im_batch = im_batch.to(self.device)

            # Get the feature vectors and feature maps from the model
            feature_vectors = self.net(im_batch, return_features=True)  # 特徴ベクトルを取得
            feature_maps = self.net(im_batch, return_featuremaps=True)  # 特徴マップを取得

            # # Normalize the feature maps (if needed)
            # feature_maps = self._normalize_feature_map(feature_maps)

            if visualize:
                # Create an empty list to hold the overlayed images
                overlayed_images = []
                for i, feature_map in enumerate(feature_maps):
                    img = im_crops[i]
                    overlayed = self.visualize_activation_map(img, feature_map)
                    overlayed_images.append(overlayed)

                # Concatenate images horizontally (you can change this to vertical concatenation if preferred)
                concatenated_img = np.concatenate(overlayed_images, axis=1)  # Horizontal concatenation

                # Convert the concatenated image to a ROS message and publish it
                ros_img = self.bridge.cv2_to_imgmsg(concatenated_img, encoding="bgr8")
                self.pub.publish(ros_img)

        return feature_vectors.cpu().numpy()

    def __call__(self, im_crops):
        if im_crops:
            features = self._get_feature(im_crops)
        else:
            features = np.array([])
        return features



if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

