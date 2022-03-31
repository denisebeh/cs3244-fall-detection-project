import cv2
from collections import deque
from Detector import FallDetector
from multiprocessing import Process

class Camera(Process):
    def __init__(self, index, realtime):
        super().__init__()
        self.detector = FallDetector()
        self.index = index
        self.realtime = realtime
        
    def run(self):
        self.cap = cv2.VideoCapture(self.index)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames = deque()

        # check if video capturing is successful
        ret, frame = self.cap.read()
        if ret:
            if self.realtime:
                print(f"Video capture successful.")
            else:
                print(f"Reading video file: '{self.index}' successful.")
        else:
            if self.realtime:
                print(f"Video capture failed.")
            else:
                print(f"Reading video file: '{self.index}' failed.")
                self.cap.release()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if ret:
                self.frames.extend(frame)

                if (len(self.frames) > self.model.config["sliding_window_length"]):
                    self.frames.popleft();

                self.detector.detect(self.frames)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()
