from states.state import State
from descriminator import Descriminator
from states.tracking_state import TrackingState
import rospy

class InitialTrainingState(State):
    def __init__(self, track_id):
        self.target_id = track_id
        self.num_pos_samples = 0
    
    def target(self):
        return self.target_id

    def state_name(self):
        return "initial_training"
    
    def update(self, descriminator: Descriminator, tracks: dict, target_position: tuple, target_InitialClassifier_count: list):
        rospy.loginfo("Initial training state")

        target_id = -1
        distance = 0.0

        for id in tracks.keys():
            pos = tracks[id].pos_in_baselink  # [x,y]
            current_dis = tracks[id].distance
            if (len(tracks.keys())!=1 and abs(pos[1])>0.2):
                continue
            if(target_id==-1 or distance > current_dis):
                target_id = id
                distance = current_dis
        
        if target_id < 0:
            return self
        
        # target_id を self.target_id に代入
        self.target_id = target_id

        isSuccessed = descriminator.updateFeatures(tracks, self.target_id)
        if isSuccessed:
            self.num_pos_samples += 1
        if self.num_pos_samples >= rospy.get_param("~initial_training_num_samples"):
            return TrackingState(self.target_id)
        return self
    

