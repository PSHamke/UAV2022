import time 

class Timer:
    duration : float = 0
    start : float = 0
    end : float = 0
    def __init__ (self):
        self.start = time.time()
    
    def End(self):
        self.end = time.time()
        self.duration = self.end- self.start;
        return self.duration

    def FPS(self):
        return float(1/(time.time() - self.start))
