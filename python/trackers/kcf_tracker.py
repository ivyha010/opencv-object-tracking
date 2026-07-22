import cv2

class KCFTracker:
    def __init__(self):
        try:
            self.tracker = cv2.legacy.TrackerKCF_create()
        except AttributeError:
            self.tracker = cv2.TrackerKCF_create()
        self.initialized = False

    def init(self, frame, bbox):
        self.tracker.init(frame, bbox)
        self.initialized = True

    def update(self, frame):
        if not self.initialized:
            raise RuntimeError("Tracker not initialized with init()")
        return self.tracker.update(frame)
