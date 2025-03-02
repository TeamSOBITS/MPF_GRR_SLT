from abc import ABC, abstractmethod

class State(ABC):
    def __init__(self):
        pass
    
    def target(self):
        return -1
    
    @abstractmethod
    def state_name(self):
        pass

    @abstractmethod
    def update(self, descriminator, tracks, target_position, target_InitialClassifier_count):
        pass
