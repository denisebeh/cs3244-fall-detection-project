from multiprocessing import Process

class Monitor(Process):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        while True:
            if not self.queue.empty():
                new_elem = self.queue.get()
                idx, result = new_elem

                # fall detected in camera {idx}
                if (result):
                    print(f"Possible fall detected in camera {idx}. Please check.")
                else:
                    print(f"Camera {idx} health status message. No falls detected.")
