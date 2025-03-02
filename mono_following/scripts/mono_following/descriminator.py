import os
import numpy as np
import rospy
# from tracklet import Tracklet
from reid.RRClassifier import RRClassifierWithStrategy
from reid.descriptor.deep.feature_extractor import PersonExtractor

"""
Thread 1:
1. Store features
2. Train Target model
Thread 2:
3. Predict the target confidence
"""
class Descriminator:
    def __init__(self):
        # Initialize the feature extraction
        # model_path = os.path.abspath(os.path.join("reid/descriptor/deep/checkpoint/ckpt.t7"))

        # model_path = "/home/sobits/catkin_ws/src/MPF_GRR_SLT/mono_following/scripts/mono_following/reid/descriptor/deep/checkpoint/ckpt.t7"
        # self.extractor = PersonExtractor(model_path=model_path)

        self.extractor = PersonExtractor()

        # Initialize our target classifier
        alpha = 1.0
        samples = 64
        threshold = 0.02
        self.classifier = RRClassifierWithStrategy(alpha, samples, threshold)

    def extractFeatures(self, tracks: dict):
        # rospy.loginfo("Extracting features")
        images = []
        idxs = []
        for idx in tracks.keys():
            if tracks[idx].image_patch is None:
                continue
            images.append(tracks[idx].image_patch)
            idxs.append(idx)
        if len(images) == 0:
            return
        descriptors = self.extractor(images)
        for i in range(len(descriptors)):
            tracks[idxs[i]].descriptor = descriptors[i]
    
    def updateFeatures(self, tracks: dict, target_id: int):
        # rospy.loginfo("Updating features")
        features = []
        labels = []
        for idx in tracks.keys():
            if tracks[idx].image_patch is None:
                continue
            # [feature, distance, distance]
            features.append([tracks[idx].descriptor, np.array(tracks[idx].distance), np.asarray([tracks[idx].region])])
            if idx == target_id:
                labels.append(1)
            else:
                labels.append(0)
        self.classifier.update_cache(features, labels)
        if len(labels)!=0:
            self.classifier.update_classifier()
            return True
        else:
            return False

    def predict(self, tracks: dict):
        # rospy.loginfo("Predicting confidences")
        for idx in tracks.keys():
            if tracks[idx].image_patch is None:
                continue
            tracks[idx].target_confidence = self.classifier.predict(tracks[idx].descriptor)

            # 各人物のidとconfidenceを出力
            rospy.loginfo(f"ID: {idx}, Confidence: {tracks[idx].target_confidence}")

    def filterFeatures(self):
        pass

    def checkIoU(self):
        pass

    def checkBox(self):
        pass
