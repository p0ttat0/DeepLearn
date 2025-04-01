import math
import time


class ProgressBar:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        time_elapsed = time.time() - self.start_time
        minutes, seconds = divmod(time_elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        print(f'\n\ntimed elapsed: {hours:.0f}:{minutes:.0f}:{seconds:.0f}')

    def update(self, epochs, batches_completed, batches_per_epoch, training_accuracy, loss, bar_length=50):
        now = time.time()
        time_elapsed = now - self.start_time
        total_batches = epochs * batches_per_epoch

        percent = (batches_completed+1) / total_batches
        filled_length = math.ceil(bar_length * percent)
        spaces_length = bar_length - filled_length

        filled = '█' * filled_length
        spaces = ' ' * spaces_length

        if percent > 0:
            time_est = (time_elapsed / percent) * (1 - percent)
            time_est_minutes, time_est_seconds = divmod(time_est, 60)
            time_est_hours, time_est_minutes = divmod(time_est_minutes, 60)

            avr_epoch_time = time_elapsed / ((batches_completed+1)/batches_per_epoch)
            avr_epoch_time_minutes, avr_epoch_time_seconds = divmod(avr_epoch_time, 60)
            avr_epoch_time_hours, avr_epoch_time_minutes = divmod(avr_epoch_time_minutes, 60)
        else:
            time_est_hours, time_est_minutes, time_est_seconds = 0, 0, 0
            avr_epoch_time_seconds, avr_epoch_time_minutes, avr_epoch_time_hours = 0, 0, 0

        print(
            f'\rprogress: [{filled}{spaces}] |{(percent * 100):.2f}%|      '
            f'loss: {loss:.7f}     '
            f'training_accuracy: {training_accuracy*100:.2f}%     '
            f'time left est: |{round(time_est_hours):02d}:{round(time_est_minutes):02d}:{round(time_est_seconds):02d}|     '
            f'average time/epoch: |{round(avr_epoch_time_hours):02d}:{round(avr_epoch_time_minutes):02d}:{round(avr_epoch_time_seconds):02d}|',
            end='')
