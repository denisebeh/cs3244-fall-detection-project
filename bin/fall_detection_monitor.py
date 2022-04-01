import multiprocessing
from source.Camera import Camera
from source.Monitor import Monitor

def single_camera_detection(idx):
    queue = multiprocessing.Queue()

    # initialize camera
    camera_0 = Camera(idx, queue)
    
    # initalize monitoring system
    monitor = Monitor(queue)

    camera_0.start()
    monitor.start()
    camera_0.join()
    monitor.join()

def multi_camera_detection(idxs):
    queue = multiprocessing.Queue()

    # initialize camera
    camera_0 = Camera(idxs[0], queue)
    camera_1 = Camera(idxs[1], queue)
    
    # initalize monitoring system
    monitor = Monitor(queue)

    camera_0.start()
    camera_1.start()
    monitor.start()
    camera_0.join()
    camera_1.join()
    monitor.join()

if __name__ == "__main__":
    """
    execute fall detection monitoring software
    """
    # launch single camera fall detection system
    single_camera_detection(0)

    # launch double camera fall detection system
    idxs = [0, 1]
    multi_camera_detection(idxs)

