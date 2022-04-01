import os
import cv2
import yaml
from collections import deque
from Detector import FallDetector
from multiprocessing import Process

class Camera(Process):
    def __init__(self, index):
        super().__init__()

        # get config
        with open('../config.yaml', "r") as f:
            self.config = yaml.safe_load(f)

        self.detector = FallDetector(self.config)
        self.index = self.config["camera_index"]
        self.w = 224
        self.h = 224
        
    def run(self):
        self.cap = cv2.VideoCapture(self.index)
        count = 0

        # check if video capturing is successful
        ret, frame = self.cap.read()
        if ret:
            print(f"Video capture on {self.index} successful.")
        else:
            print(f"Video capture on {self.index} failed.")
            self.cap.release()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if ret:
                count += 1
                path = self.config["output_path"] + "raw_img_" + str(count);
                flow_x = self.config["flow_path"] + "flow_x/flow_img_" + str(count)
                flow_y = self.config["flow_path"] + "flow_y/flow_img_" + str(count)
                cv2.resize(frame, (self.w, self.h))
                cv2.imwrite(frame, path)
                os.system('/home/denise/dense_flow/build/extract_cpu -f={} -x={} -y={} -i=tmp/image -b=20 -t=1 -d=0 -s=1 -o=dir'.format(path, flow_x, flow_y))
                
                if count > 10:
                    self.detector.detect(count)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()