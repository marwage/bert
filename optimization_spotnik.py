# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from kungfu._utils import map_maybe
from kungfu.tensorflow.ops import (group_all_reduce, spotnik_group_all_reduce, spotnik_all_reduce, peer_info, spotnik_request_variable_with_template, request_variable_with_template)


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu, threshold):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  
  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)

  # calculate Exponential moving average
  def exponential_moving_average_loss_fn(loss, alpha = 0.1):
    exponential_moving_average_loss = tf.Variable(8.0, name="exponential_moving_average_loss")
    sum_loss, succeeded = spotnik_all_reduce(loss)
    avg_loss = tf.math.divide(sum_loss, np)
    assign_op = tf.assign(exponential_moving_average_loss, alpha * avg_loss + (1-alpha) * exponential_moving_average_loss)
    tf.summary.scalar("exponential_moving_average_loss", exponential_moving_average_loss)
    return tf.cond(tf.math.equal(succeeded, 0),
      lambda: assign_op,
      lambda: tf.identity(exponential_moving_average_loss))

  # KungFu
  # add averaging over all gradient
  rank, num_workers = peer_info()
  np = tf.cast(num_workers, tf.float32)

  def s_sgd_fn(optimizer, variables, grads, num_workers):
    summed_grads, num_unsucceeded = spotnik_group_all_reduce(grads)
    grads = map_maybe(lambda g: g / num_workers, summed_grads)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    return tf.cond(tf.math.greater(num_unsucceeded, 0),
      lambda: tf.no_op(),
      lambda: optimizer.apply_gradients(
        zip(grads, variables), global_step=global_step))

  def vanilla_s_sgd_fn(optimizer, vars, grads, num_workers):
    summed_grads = group_all_reduce(grads)
    grads = map_maybe(lambda g: g / num_workers, summed_grads)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    return optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

  def sma_fn(optimizer, variables, grads, num_workers, alpha=0.1):
    summed_vars, num_unsucceeded = spotnik_group_all_reduce(variables)
    averaged_vars = [v / num_workers for v in summed_vars]
    assign_ops = [
      tf.assign(v, (1 - alpha) * v + alpha * avg_v)
      for v, avg_v in zip(variables, averaged_vars)
    ]
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    with tf.control_dependencies(assign_ops):
      return tf.cond(tf.math.greater(num_unsucceeded, 0),
      lambda: tf.no_op(),
      lambda: optimizer.apply_gradients(
          zip(grads, variables), global_step=global_step))

  def get_random_peer(cluster_size, self_rank):
    t = tf.random.uniform([], minval=0, maxval=cluster_size, dtype=tf.int32)
    return tf.cond(tf.equal(t, self_rank),
      lambda: tf.math.floormod(t + 1, cluster_size),
      lambda: tf.identity(t))

  def pair_fn(optimizer, variables, grads, num_workers, rank):
    target = get_random_peer(num_workers, rank)
    other_peer = [spotnik_request_variable_with_template(target, v) for v in variables]
    other_peer_vars = [t[0] if t is not None else None for t in other_peer]
    not_succeeded = [t[1] for t in other_peer if t is not None]
    num_not_succeeded = tf.add_n(not_succeeded)
    assign_ops = [
        tf.assign(v, 0.5 * (v + other_v))
        for v, other_v in zip(variables, other_peer_vars)
    ]

    with tf.control_dependencies(assign_ops):
      return tf.cond(tf.math.greater(num_not_succeeded, 0),
        lambda: tf.no_op(),
        lambda: optimizer.apply_gradients(zip(grads, variables), global_step=global_step))

  def vanilla_pair_fn(optimizer, variables, grads, num_workers, rank):
    target = get_random_peer(num_workers, rank)
    other_peer = [request_variable_with_template(target, v) for v in variables]
    assign_ops = [
        tf.assign(v, 0.5 * (v + other_v))
        for v, other_v in zip(variables, other_peer)
    ]

    with tf.control_dependencies(assign_ops):
      return optimizer.apply_gradients(zip(grads, variables), global_step=global_step)
  
  # train_op = tf.case([(tf.math.less_equal(np, 2), lambda: s_sgd_fn(optimizer, tvars, grads, np)),
  #   (tf.math.less_equal(np, 8), lambda: sma_fn(optimizer, tvars, grads, np))],
  #         default=lambda: pair_fn(optimizer, tvars, grads, num_workers, rank), exclusive=True)
  with tf.control_dependencies([exponential_moving_average_loss_fn(loss)]):
    train_op = pair_fn(optimizer, tvars, grads, num_workers, rank)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
  # a different optimizer, you should probably take this line out.
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      # KungFu
      # add reuse=tf.AUTO_REUSE for the Adaptive SGD
      with tf.variable_scope("apply_gradients", reuse=tf.AUTO_REUSE):
        m = tf.get_variable(
            name=param_name + "/adam_m",
            shape=param.shape.as_list(),
            dtype=tf.float32,
            trainable=False,
            initializer=tf.zeros_initializer())
        v = tf.get_variable(
            name=param_name + "/adam_v",
            shape=param.shape.as_list(),
            dtype=tf.float32,
            trainable=False,
            initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
