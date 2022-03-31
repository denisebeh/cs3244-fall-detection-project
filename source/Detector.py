from Model import Model

class FallDetector:
    def __init__(self):
        self.model = Model()

    def detect(self, frames):
        # optical flow pipeline

        # model prediction pipeline