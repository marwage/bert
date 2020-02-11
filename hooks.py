import tensorflow as tf
import time
from kungfu import current_cluster_size


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


class ScalingFactorHook(tf.train.SessionRunHook):
  def __init__(self, batch_size):
    super(ScalingFactorHook, self).__init__()
    self.batch_size = batch_size
    self.time_difference = 0.0
    self.examples_per_second = dict()

  def begin(self):
    self.last_time_stamp = time.time()

  def after_run(self, run_context, run_values):
    now = time.time()
    self.time_difference = now - self.last_time_stamp
    self.last_time_stamp = now
    size = current_cluster_size()
    if size in self.examples_per_second:
      beta = 0.1
      self.examples_per_second[size] = (1-beta) * self.examples_per_second[size] + (beta * (self.batch_size * size) / self.time_difference)
    else:
      self.examples_per_second[size] = (self.batch_size * size) / self.time_difference
    print("global examples per second ", self.examples_per_second)
    

class LossDeltaHook(tf.train.SessionRunHook):
  def __init__(self):
    super(LossDeltaHook, self).__init__()

  def calc(self):
    loss_delta = tf.math.subtract(self.loss, self.last_loss)
    assign_loss_delta = tf.cond(tf.math.equal(self.global_step, 0),
      lambda: tf.identity(self.loss_delta),
      lambda: tf.assign(self.loss_delta, loss_delta))

    beta = 0.1
    ma_loss_delta = tf.math.add(self.ma_loss_delta * (1-beta), beta * self.loss_delta)
    assign_ma_loss_delta = tf.assign(self.ma_loss_delta, ma_loss_delta)

    with tf.control_dependencies([assign_loss_delta]):
      with tf.control_dependencies([assign_ma_loss_delta]):
        return tf.assign(self.last_loss, self.loss)
    
  def begin(self):
    self.loss = tf.get_default_graph().get_tensor_by_name("average_loss:0")
    self.global_step = tf.train.get_or_create_global_step()
    self.last_loss = tf.get_variable("last_loss", initializer=0.0)
    self.ma_loss_delta = tf.get_variable("ma_loss_delta", initializer=0.0)
    self.loss_delta = tf.get_variable("loss_delta", initializer=0.0)
    tf.summary.scalar("ma_loss_delta", self.ma_loss_delta)
    tf.summary.scalar("loss_delta", self.loss_delta)

    self.calc_op = self.calc()

  def after_run(self, run_context, run_values):
    run_context.session.run(self.calc_op)
