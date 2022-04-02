import gc
import cv2
import numpy as np
import scipy.io as sio
from .Model import Model

class FallDetector:
    def __init__(self, config):
        self.model = Model(config)
        self.config = config

    def detect(self, end_idx):
        # optical flow pipeline
        features = self.get_features(end_idx)

        # model prediction pipeline
        return self.model.predict(features)

    def get_features(self, end_idx):
        """
        method containing optical flow processing pipline on sliding window frames
        """
        #get sliding window frames
        x_frames = []
        y_frames = []
        d = sio.loadmat(self.config["mean_file"])
        flow_mean = d['image_mean']
        flow = np.zeros(shape=(224, 224, 2 * self.config["sliding_window_length"], 1), dtype=np.float64)

        for i in range(self.config["sliding_window_length"]):
            flow_x_file = self.config["flow_path"] + "flow_x/flow_img_" + str(end_idx)
            flow_y_file = self.config["flow_path"] + "flow_y/flow_img_" + str(end_idx)
            img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)

            # Assign an image i to the jth stack in the kth position, but also
            # in the j+1th stack in the k+1th position and so on	
            # (for sliding window) 
            for s in list(reversed(range(min(10,i+1)))):
                if i-s < 1:
                    flow[:,:,2*s,  i-s] = img_x
                    flow[:,:,2*s+1,i-s] = img_y
            
            del img_x,img_y
            gc.collect()

            end_idx -= 1
            if end_idx < 0:
                end_idx = self.config["sliding_window_length"] - 1

        # Subtract mean
        flow = flow - np.tile(flow_mean[..., np.newaxis], (1, 1, 1, flow.shape[3]))
        flow = np.transpose(flow, (3, 0, 1, 2))
        predictions = np.zeros((flow.shape[0], self.config("num_features")), dtype=np.float64)

        # Process each stack: do the feed-forward pass
        for i in range(flow.shape[0]):
            prediction = self.model.model.predict(np.expand_dims(flow[i, ...], 0))
            predictions[i, ...] = prediction
            
        return predictions
