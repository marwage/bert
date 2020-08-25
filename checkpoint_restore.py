import tensorflow as tf
import re
import time

class CheckpointAndRestoreHook(tf.train.SessionRunHook):
    def __init__(self):
        self._checkpoints = dict()
        self._checkpoint_placeholers = []
        self._restore_checkpoint = False
        self._branch = 0

        # intermediately
        self._checkpoint_key = (100,0)

    def begin(self):
        self._global_variables = tf.global_variables()
        for var in self._global_variables:
            m = re.match("^.+?(?=:0)", var.name)
            var_name = m.group(0)
            var_name = var_name + "_ckpt"
            self._checkpoint_placeholers.append(tf.placeholder(var.dtype, shape=var.shape, name=var_name))
        self._restore_op = [tf.assign(var, ckpt) for var, ckpt in zip(self._global_variables, self._checkpoint_placeholers)]

    def after_run(self, run_context, run_values):
        global_step = run_context.session.run(tf.train.get_or_create_global_step())
        if global_step % 100 == 0:
            begin = time.time()
            self._checkpoints[(global_step, self._branch)] = run_context.session.run(self._global_variables)
            duration = time.time() - begin
            print("checkpoint duration", duration)

    def before_run(self, run_context):
        global_step = run_context.session.run(tf.train.get_or_create_global_step())
        if global_step == 150:
            self._restore_checkpoint = True
        else:
            self._restore_checkpoint = False
        if self._restore_checkpoint:
            begin = time.time()
            run_context.session.run(self._restore_op, feed_dict={ckpt_pl.name: ckpt for ckpt_pl, ckpt in zip(self._checkpoint_placeholers, self._checkpoints[self._checkpoint_key])})
            duration = time.time() - begin
            print("restore duration", duration)
            self._branch = self._branch + 1
            print("restored checkpoint at ", global_step)
            
            run_context.request_stop()
