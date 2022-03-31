from Model import Model

class FallDetector:
    def __init__(self):
        self.model = Model()

    def detect(self, frames):
        # optical flow pipeline
        features = self.get_features(frames)

        # model prediction pipeline
        self.model.predict(features)

    def get_features(self, frames):
        """
        method containing optical flow processing pipline on sliding window frames
        """
        return

    def process_ufrd(self):
        """
        method containing pipeline to train model with UFRD dataset
        """
        return

    def process_combined(self):
        """
        method containing pipeline to train model with combined datasets
        """
        return