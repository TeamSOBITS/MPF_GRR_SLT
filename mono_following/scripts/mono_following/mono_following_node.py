#! /usr/bin/env python3
import numpy as np
import ros_numpy
import cv2
from tqdm import main
import rospy

# 追加
import message_filters
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# My class
from tracklet import Tracklet
from states.initial_state import InitialState
from descriminator import Descriminator
# standard ROS message
from sensor_msgs.msg import Image
# import tf

# my ROS message
from mono_tracking.msg import TrackArray
from mono_tracking.msg import Track
from mono_following.msg import Target
# from mono_following.msg import Box

#追加
from sobits_msgs.msg import BoundingBoxes, ObjectPoseArray


class MonoFollowing:
    def __init__(self):
        ### Some parameters for debug and testing ###
        self.SAVE_TRACKS = False
        self.SAVE_IMAGES = False
        
        # self.IMAGE_PATH = "/home/jing/Data/Dataset/FOLLOWING/ccf_failure_dataset_masaic/ccf_failure_dataset/room433/debug/grr_slt"
        # self.STORE_DIR = rospy.get_param("~track_store_dir")

        ### Our parameters ###
        self.state = InitialState()
        self.descriminator = Descriminator()

        self.previous_time_ = 0.0

        self.target_position = None  
        self.target_InitialClassifier_count = [0]  # 初期化

        ### Our necessary topics ###
        # listen to the transform tree
        # self.tf_listener = tf.TransformListener()

        # message_filtersのSubscriberを設定
        self.bb_sub = message_filters.Subscriber("/yolov10_bbox_to_tf/object_rects", BoundingBoxes)
        self.op_sub = message_filters.Subscriber("/yolov10_bbox_to_tf/object_poses", ObjectPoseArray)
        self.image_sub = message_filters.Subscriber("/rgb/image_raw", Image)

        # tsをインスタンス変数にする
        self.ts = message_filters.ApproximateTimeSynchronizer([self.bb_sub, self.op_sub, self.image_sub], 10, 0.1)
        self.ts.registerCallback(self.callback)

        # # subscribe tracks from mono_tracking
        # self.tracks_sub = rospy.Subscriber("/mono_tracking/tracks", TrackArray, self.callback)

        # publish image for visualization
        self.image_pub = rospy.Publisher("/mono_following/vis_image", Image, queue_size=1)
        # publish image patches for visualization
        self.patches_pub = rospy.Publisher("/mono_following/patches", Image, queue_size=1)
        # publish target information
        self.target_pub = rospy.Publisher("/mono_following/target", Target, queue_size=1)

        ### Node is already set up ###
        rospy.loginfo("Mono Following Node is Ready!")
        rospy.spin()

    def preprocessImage(self, image, patch):
        ### prior parameters ### 
        self.down_width = 128  #64
        self.down_height = 320  #128

        image = image[patch[1]:patch[3], patch[0]:patch[2], :]
        image = cv2.resize(image, (self.down_width, self.down_height), interpolation=cv2.INTER_LINEAR)
        return image
    
    # def callback(self, tracks_msg):
    def callback(self, bb_msg, op_msg, img_msg):
        # print("CALLBACK")
        # get messages
        # self.tracks = {}

        dt = img_msg.header.stamp.to_sec() - self.previous_time_  # 前回のタイムスタンプとの差分を計算
        self.previous_time_ = img_msg.header.stamp.to_sec()
        rospy.loginfo(f"Loop time (dt): {dt:.6f} seconds!!!!!!!!!!!!!!!!!!!!!!!")

        # Convert ROS image message to OpenCV format using ros_numpy
        try:
            img_raw = ros_numpy.numpify(img_msg)[:,:,[2,1,0]]  # BGRをRGBに変換、（ここではあまり関係ないが、その際付属のAlphaチャンネル（透明度）は省かれる（bgra->rgb））
        except Exception as e:
            rospy.logerr(f"ros_numpy exception: {str(e)}")
            return

        # Check if image is empty
        if img_raw is None:
            rospy.logerr("input_img error")
            return

        # Check bounding box and object pose availability
        check_bb = bb_msg.bounding_boxes[0].probability if len(bb_msg.bounding_boxes) > 0 else None
        check_op = op_msg.object_poses[0].pose.position.x if len(op_msg.object_poses) > 0 else None

        if check_bb is None:
            rospy.logerr("bb_msg not found")
            return
        if check_op is None:
            rospy.logerr("op_msg not found")
            return

        self.tracks = {}

        # Check if bounding_boxes and object_poses have the same length
        if len(bb_msg.bounding_boxes) != len(op_msg.object_poses):
            rospy.logerr(f"Size mismatch: bounding_boxes({len(bb_msg.bounding_boxes)}) != object_poses({len(op_msg.object_poses)})")
            return

        # bounding_boxes を xmin の昇順にソート（毎フレーム、画像の左側から右側の順番に並べ替え）
        sorted_indices = sorted(range(len(bb_msg.bounding_boxes)), key=lambda i: bb_msg.bounding_boxes[i].xmin)

        # ソート後のリストを作成
        bb_msg.bounding_boxes = [bb_msg.bounding_boxes[i] for i in sorted_indices]
        op_msg.object_poses = [op_msg.object_poses[i] for i in sorted_indices]

        id = 0
        for op in op_msg.object_poses:
            self.tracks[id] = Tracklet(img_msg.header, op, img_raw)
            id += 1

        id = 0
        for bb in bb_msg.bounding_boxes:

            region_xmin = bb.xmin
            region_ymin = bb.ymin
            region_xmax = bb.xmax
            region_ymax = bb.ymax

            person_region = [region_xmin, region_ymin, region_xmax, region_ymax]

            self.tracks[id].region = person_region

            image_patch = self.preprocessImage(img_raw, person_region)

            self.tracks[id].image_patch = image_patch
            id += 1

        # extract features
        self.descriminator.extractFeatures(self.tracks)
        
        # update the state
        # print(self.state.state_name())
        next_state = self.state.update(self.descriminator, self.tracks, self.target_position, self.target_InitialClassifier_count)
        if next_state is not self.state:
            self.state = next_state

        target_id_keep = self.state.target()

        if target_id_keep in self.tracks:
            
            target_position_keep = self.tracks[target_id_keep]
            self.target_position = target_position_keep.pos_in_baselink  

            # Calculate the Euclidean distance
            distance = np.linalg.norm(self.target_position)
            rospy.loginfo(f"Previous Target ID: {target_id_keep}, Previous Target Distance: {distance:.2f}")
        else:
            rospy.logwarn(f"Previous Target ID: {target_id_keep} not found in tracks")
        
        
        # publish target information
        if self.target_pub.get_num_connections():
            target = Target()
            target.header = img_msg.header
            target.state.data = self.state.state_name()
            target.target_id = self.state.target()
            
            track_ids = []
            confidences = []
            for idx in self.tracks.keys():
                track_ids.append(idx)
                if idx == target.target_id and self.tracks[idx].target_confidence != None:
                    target.box = self.tracks[idx].region # u_tl, v_tl, u_br, v_br
                    target.classifier_confidences = [self.tracks[idx].target_confidence]
                    target.position = self.tracks[idx].pos_in_baselink
                if self.tracks[idx].target_confidence != None:
                    confidences.append(self.tracks[idx].target_confidence)
            target.track_ids = track_ids
            target.confidences = confidences
            self.target_pub.publish(target)
        
        # publish image containing the identification result
        if self.image_pub.get_num_connections():
            image_bgr = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
            # # アルファチャンネルを削除する
            # if image_bgr.shape[2] == 4:
            #     image_bgr = image_bgr[:, :, :3]
            
            image_bgr = self.visualize(image_bgr, self.tracks)
            image_msg = ros_numpy.msgify(Image, image_bgr, encoding = "bgr8")
            self.image_pub.publish(image_msg)
            if self.SAVE_IMAGES:
                pass

        # publish target patch to see whether is OK
        if self.patches_pub.get_num_connections():
            image = self.visualize_patches(self.tracks)
            if image is not None:
                # cv2.imwrite("./test.jpg", image)

                # # アルファチャンネルを削除する
                # if image.shape[2] == 4:
                #     image = image[:, :, :3]
                    
                image_msg = ros_numpy.msgify(Image, image, encoding = "rgb8")
                self.patches_pub.publish(image_msg)

        # Save tracks information, for debug and analysis
        if self.SAVE_TRACKS:
            pass

    def visualize(self, original_image, tracks):
        # print("VISUALIZE")
        image = original_image.copy()
        for idx in tracks.keys():
            if tracks[idx].target_confidence == None:
                continue
            image = cv2.putText(original_image, self.state.state_name(), (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            # 信頼値とIDを表示するラベルを作成
            confidence = tracks[idx].target_confidence
            label = f"id:{idx}, conf:{confidence:.4f}"

            # ラベルのサイズを取得
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, 0.75, 1)

            # ラベルの位置を各バウンディングボックスの左上に配置
            x = self.tracks[idx].region[0]
            y = self.tracks[idx].region[1]
            label_rect = (x, y, label_size[0], label_size[1])

            # ラベルの背景（白い四角形）を描画 (label_rectを使用)
            cv2.rectangle(image, (label_rect[0], label_rect[1]), 
                        (label_rect[0] + label_rect[2], label_rect[1] + label_rect[3]), 
                        (255, 255, 255), -1)

            # ラベルのテキストを描画
            image = cv2.putText(image, label, (x, y + label_rect[3]), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 0), 1)

            if idx == self.state.target():
                image = cv2.rectangle(image, (self.tracks[idx].region[0],self.tracks[idx].region[1]), (self.tracks[idx].region[2],self.tracks[idx].region[3]), (0,255,0), 2)
                # image = cv2.putText(image, "id:{:d}".format(idx), (int((self.tracks[idx].region[0]+self.tracks[idx].region[2])/2-5), int((self.tracks[idx].region[1]+self.tracks[idx].region[3])/2-5)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
            else:
                image = cv2.rectangle(image, (self.tracks[idx].region[0],self.tracks[idx].region[1]), (self.tracks[idx].region[2],self.tracks[idx].region[3]), (0,0,255), 2)
                # image = cv2.putText(image, "id:{:d}".format(idx), ((int((self.tracks[idx].region[0]+self.tracks[idx].region[2])/2-5), int((self.tracks[idx].region[1]+self.tracks[idx].region[3])/2-5))), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
        return image
    
    def saveTrack(self, track_id, track, store_path):
        print("SAVE TRACKS")

    def visualize_patches(self, tracks):
        target_id = self.state.target()
        if target_id in tracks.keys() and self.tracks[target_id].image_patch is not None:
            return self.tracks[target_id].image_patch
        else:
            return None
if __name__ == "__main__":
    rospy.init_node('mono_following', anonymous=True)
    mono_following = MonoFollowing()

