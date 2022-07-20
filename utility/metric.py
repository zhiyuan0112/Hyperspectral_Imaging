import numpy as np

from .indexes import cal_bwpsnr, cal_bwssim, cal_sam


def PSNR(outputs, targets):
    return np.mean(cal_bwpsnr(outputs, targets))

def SSIM(outputs, targets):
    return np.mean(cal_bwssim(outputs, targets))

def SAM(outputs, targets):
    return cal_sam(outputs, targets)


class MetricTracker:
    def __init__(self, metrics:dict):
        self.metrics = metrics
        self.refresh()

    def refresh(self):
        self.values = {name: [] for name in self.metrics.keys()}
        self.count = 0
    
    def update(self, outputs, targets):
        for name, metric_func in self.metrics.items():
            metric = metric_func(outputs, targets)
            self.values[name].append(metric)
        self.count += 1
    
    def get(self, key):
        return sum(self.values[key]) / self.count

    def get_all(self):
        avg_values = {name: [] for name in self.metrics.keys()}
        for name, value in self.values.items():
            avg_values[name] = sum(value) / self.count
        return avg_values