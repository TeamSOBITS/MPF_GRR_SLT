from states.state import State
import sys
sys.path.append("..")
from descriminator import Descriminator
import rospy

import math

class TrackingState(State):
    def __init__(self, target_id):
        self.target_id = target_id
    
    def target(self):
        return self.target_id
        
    def state_name(self):
        return "tracking"
    
    def update(self, descriminator: Descriminator, tracks: dict, target_position: tuple, target_InitialClassifier_count: list):
        rospy.loginfo("tracking state")
        
        from states.reid_state import ReidState
        
        # まず、各トラックのpred(信頼度)を計算
        descriminator.predict(tracks)

        # 距離ベースで最も近い人物IDを取得
        min_distance = float('inf')
        closest_target_id = -1
        DistanceBase_targetID = False

        # 対象者のみの状態から、非対象者も加わって初期分類が行われる際、誤識別を防ぐための補助処理
        if len(tracks) >= 2 and target_InitialClassifier_count[0] < 5:
            for track_id, track in tracks.items():
                pose_x = track.pos_in_baselink[0]
                pose_y = track.pos_in_baselink[1]

                # 距離を計算
                distance = math.hypot(pose_x - target_position[0], pose_y - target_position[1])

                # 各trackのdistanceとtrack_idを可視化
                rospy.loginfo(f"Track ID: {track_id}, Distance: {distance}, Pose (x, y): ({pose_x}, {pose_y})")

                if distance < min_distance:
                    min_distance = distance
                    closest_target_id = track_id

            DistanceBase_targetID = True
            target_InitialClassifier_count[0] += 1

        # 最大のpredとそれに対応するtarget_idを探す
        max_pred = float('-inf')
        best_target_id = -1

        for track_id, track in tracks.items():
            if DistanceBase_targetID:
                # closest_target_idを基にpredを取得
                if track_id == closest_target_id:
                    pred = track.target_confidence
                    max_pred = pred
                    best_target_id = track_id
                    break
            else:
                pred = track.target_confidence  # 各トラックの信頼度を取得
                if pred is not None and pred > max_pred:
                    max_pred = pred
                    best_target_id = track_id

        # best_target_idが見つからない場合、ReidStateに遷移
        if best_target_id == -1:
            rospy.loginfo("No valid target found, switching to ReidState.")
            return ReidState()

        # IDスイッチの検出（最大のpredが閾値を下回った場合）
        if max_pred < rospy.get_param('~id_switch_detection_thresh', default=-1.0):
            rospy.loginfo("ID switch detected!!")
            return ReidState()

        # # ターゲットが見えなくなった場合の処理
        # if max_pred < rospy.get_param("~min_target_confidence", -1):
        #     rospy.loginfo("do not update for pred < min_target_confidence")
        #     return self

        # predが最大のtarget_idを使って特徴量を更新
        isSuccessed = descriminator.updateFeatures(tracks, best_target_id)

        # 更新が成功したかどうかにかかわらず、TrackingStateのまま維持
        self.target_id = best_target_id  # 最大のtarget_idに更新

        return self
    

