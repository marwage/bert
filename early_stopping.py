import tensorflow as tf
import time

class EarlyStoppingHook(tf.train.SessionRunHook):
    def __init__(self):
        self._start = time.time()
        self._duration = 0.5 * 60 * 60

    def after_run(self, run_context, run_values):
        if time.time() - self._start > self._duration:
            print("stopping")
            run_context.request_stop()
