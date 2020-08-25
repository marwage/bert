import tensorflow as tf
import time
import json
import numpy as np
import urllib
import os
from kungfu import current_rank

class ScalingHook(tf.train.SessionRunHook):
    def __init__(self, batch_size):
        self._batch_size = batch_size
        self._throughputs = []
        self._average_throughput = 0
        self._single_worker_throughput = 0
        path = os.path.join(os.environ["HOME"], "playground/workers.json")
        self._workers = self.read_workers(path)
        self._stop_scaling = False
        self._current_num_workers = 1

    def begin(self):
        self._global_step = tf.train.get_or_create_global_step()

    def before_run(self, run_context):
        if current_rank() == 0:
            self._start_time = time.time()

    def after_run(self, run_context, run_values):
        if current_rank() == 0:
            duration = time.time() - self._start_time
            global_step = run_context.session.run(self._global_step)
            change_step = 20
            if global_step % change_step == 0:
                if global_step == change_step: # only one worker
                    self._single_worker_throughput = np.mean(self._throughputs)
                else:
                    if self._average_throughput < self._single_worker_throughput / 2:
                        self.remove_last_worker()
                        self._stop_scaling = True
                    else:
                        if not self._stop_scaling:
                            if self._current_num_workers < len(self._workers):
                                self.add_worker()
                            else:
                                self._stop_scaling = True
                self._throughputs = []
                self.average_throughput = 0
            else:
                self._throughputs.append(self._batch_size / duration)
                self._average_throughput = np.mean(self._throughputs)
                print("global_step", global_step, "average_throughput", self._average_throughput)

    def read_workers(self, path):
        with open(path, "r") as json_file:
            data = json_file.read()
        return json.loads(data)

    def add_worker(self, url="http://127.0.0.1:9100/addworker"):
        worker = self._workers[self._current_num_workers - 1]
        print(worker)
        
        data = json.dumps(worker).encode("utf-8")
        req =  urllib.request.Request(url, data=data, method="POST")
        resp = urllib.request.urlopen(req)
        if resp.getcode() != 200:
            print("request failed")
            return
        print("success")
        self._current_num_workers = self._current_num_workers + 1

    def remove_last_worker(self, url="http://127.0.0.1:9100/removeworker"):
        try:
            worker = self._workers[self._current_num_workers - 2]
        except:
            print("cannot remove the only worker")
            return
        data = urllib.parse.urlencode(worker).encode()
        req =  urllib.request.Request(url, data=data)
        resp = urllib.request.urlopen(req)
        if resp.getcode() != 200:
            print("request failed")
            return
        self._current_num_workers = self._current_num_workers - 1


if __name__ == "__main__":
    batch_size = 8
    scaling_hook = ScalingHook(batch_size)
    scaling_hook.add_worker()
    url = "http://127.0.0.1:9100/get"
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req)
    print(json.loads(resp.read()))