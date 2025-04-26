import numpy as np
from matplotlib import pyplot as plt


class MetricTracker:
    def __init__(self, model, tracked: list):
        # model performance
        self.training_accuracy = []
        self.training_losses = []

        # training metrics
        self.gradient_magnitude = []
        self.gradient_extremes = []
        self.activation_magnitude = []
        self.activation_extremes = []

        self.model = model
        self.tracked_metrics = tracked
        self.relevant_layers = sum([1 for layer in model.layers if layer.type not in ['reshape', 'dropout']])

    def training_metrics_update(self, activations_magnitude, activations_extremes, gradient_magnitude, gradient_extremes):
        self.gradient_magnitude.append(gradient_magnitude)
        self.gradient_extremes.append(gradient_extremes)
        self.activation_magnitude.append(activations_magnitude)
        self.activation_extremes.append(activations_extremes)

    def performance_metrics_update(self, loss, training_accuracy):
        self.training_accuracy.append(training_accuracy)
        self.training_losses.append(loss)

    def show(self):
        temp = {'training accuracy': self.training_accuracy,
                'training losses': self.training_losses,
                'gradient magnitude': self.gradient_magnitude,
                'gradient extremes': self.gradient_extremes,
                'activation magnitude': self.activation_magnitude,
                'activation extremes': self.activation_extremes
                }
        tracked = [temp[metric] for metric in self.tracked_metrics if metric in temp]

        assert len(tracked) == len(self.tracked_metrics)

        for i in range(len(self.tracked_metrics)):
            x, y = np.arange(len(tracked[i])), tracked[i]
            plt.plot(x, y, 'o')
            if len(tracked[i]) > 1:
                plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
            plt.title(self.tracked_metrics[i])
            plt.show()

    def reset(self):
        self.training_accuracy = []
        self.training_losses = []
        self.gradient_magnitude = []
        self.gradient_extremes = []
        self.activation_magnitude = []
