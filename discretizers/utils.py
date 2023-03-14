import math

def sign(val):
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return -1

class DiscreteRange:
    def __init__(self, start, end, num_intervals):
        self.start = start
        self.end = end
        self.num_intervals = num_intervals

    def __call__(self, val):
        if val < self.start:
            return 0
        elif val > self.end:
            return self.num_intervals
        else:
            return int((val - self.start) / (self.end - self.start) * self.num_intervals)

class DiscreteInf:
    def __init__(self, num_intervals, factor):
        self.num_intervals = num_intervals
        self.factor = factor

    def sqrt_smooth(self, val):
        if val == 0:
            return 0
        
        return sign(val) * math.sqrt(abs(val)) * self.factor
        # return val * self.factor

    def __call__(self, val):
        if val == 0:
            return 0

        v = int(self.sqrt_smooth(val))

        v = max(- self.num_intervals // 2, v)

        v = min(self.num_intervals // 2, v)

        return v




