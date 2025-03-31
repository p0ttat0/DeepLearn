import numpy as np
from matplotlib import pyplot as plt


class MetricTracker:
    def __init__(self, model, tracked: list):
        self.training_accuracy = []
        self.training_losses = []

        self.gradient_magnitude = []
        self.gradient_magnitude_cache = []
        self.gradient_extremes = []
        self.gradient_extremes_cache = []
        self.activation_magnitude = []
        self.activation_magnitude_cache = []

        self.model = model
        self.tracked = tracked

    def bp_metrics_update(self, output_gradient, activations):
        if 'gradient magnitude' in self.tracked:
            self.gradient_magnitude_cache.append(np.mean(np.abs(output_gradient)))
            if len(self.gradient_magnitude_cache) == len(self.model.layers):
                self.gradient_magnitude.append(np.mean(self.gradient_magnitude_cache))
                self.gradient_magnitude_cache = []
        if 'gradient extremes' in self.tracked:
            self.gradient_extremes_cache.append(np.max(output_gradient) + np.abs(np.min(output_gradient)) / 2)
            if len(self.gradient_extremes_cache) == len(self.model.layers):
                self.gradient_extremes.append(np.mean(self.gradient_extremes_cache))
                self.gradient_extremes_cache = []
        if 'activation magnitude' in self.tracked:
            self.activation_magnitude_cache.append(np.mean(np.abs(activations)))
            if len(self.activation_magnitude_cache) == len(self.model.layers):
                self.activation_magnitude.append(np.mean(self.activation_magnitude_cache))
                self.activation_magnitude_cache = []

    def performance_metrics_update(self, loss, training_accuracy):
        self.training_accuracy.append(training_accuracy)
        self.training_losses.append(loss)

    def show(self):
        temp = {'training accuracy': self.training_accuracy,
                'training losses': self.training_losses,
                'gradient magnitude': self.gradient_magnitude,
                'gradient extremes': self.gradient_extremes,
                'activation magnitude': self.activation_magnitude
                }
        tracked = [temp[metric] for metric in self.tracked if metric in temp]

        assert len(tracked) == len(self.tracked)

        for i in range(len(self.tracked)):
            x, y = np.arange(len(tracked[i])), tracked[i]
            plt.plot(x, y, 'o')
            if len(tracked[i]) > 1:
                plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
            plt.title(self.tracked[i])
            plt.show()

    def reset(self):
        self.training_accuracy = []
        self.training_losses = []
        self.gradient_magnitude = []
        self.gradient_extremes = []
        self.activation_magnitude = []
