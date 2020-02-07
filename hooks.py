import tensorflow as tf
import time


class AdaSGDHook(tf.train.SessionRunHook):
  def __init__(self):
    super(AdaSGDHook, self).__init__()
    self._changed_yet = False
    self._change_step = 4000
  
  def begin(self):
    from kungfu.tensorflow.ops import broadcast
    self._ops = [tf.assign(v, broadcast(v)) for v in tf.global_variables()]
    self._s_sgd = tf.get_default_graph().get_tensor_by_name("s_sgd:0")
    self._assign_op = tf.assign(self._s_sgd, 1)
    self._global_step = tf.train.get_or_create_global_step()

  def after_run(self, run_context, run_values):
    global_step = run_context.session.run(self._global_step)
    if global_step >= self._change_step and not self._changed_yet:
        run_context.session.run(self._ops)
        run_context.session.run(self._assign_op)
        self._changed_yet = True
        print("Changed at step ", global_step)


class EarlyStoppingHook(tf.train.SessionRunHook):
  def __init__(self, threshold):
    super(EarlyStoppingHook, self).__init__()
    self._loss_threshold = threshold
    self._twice = False
    
  def begin(self):
    self._loss = tf.get_default_graph().get_tensor_by_name("average_loss:0")

  def after_run(self, run_context, run_values):
    loss = run_context.session.run(self._loss)
    print("average_loss:0, %d", loss)

    if loss < self._loss_threshold and self._twice:
      print("loss below threshold")
      print("stopping")
      run_context.request_stop()
    
    if loss < self._loss_threshold and not self._twice:
      self._twice = True
    else: 
      self._twice = False


class PrintHook(tf.train.SessionRunHook):
  def __init__(self):
    super(PrintHook, self).__init__()
    
  def begin(self):
    self._variables = tf.global_variables()

  def after_run(self, run_context, run_values):
    for var in self._variables:
      print(var.name)


class TimingHook(tf.train.SessionRunHook):
  def __init__(self):
    super(TimingHook, self).__init__()
    
  def begin(self):
    self.last_time_stamp = time.time()

  def after_run(self, run_context, run_values):
    now = time.time()
    time_difference = now - self.last_time_stamp
    print("last iteration in ", time_difference, " seconds")
    self.last_time_stamp = now
