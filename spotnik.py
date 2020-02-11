import os
import time

import numpy as np

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.training import training_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import meta_graph
from tensorflow.core.util.event_pb2 import SessionLog

from kungfu import current_cluster_size
from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
from kungfu.tensorflow.ops import _get_init_step, counter, resize_cluster, step_based_schedule


class CheckpointSaverHook():
    def __init__(self,
                checkpoint_dir,
                saver=None,
                checkpoint_basename="model.ckpt",
                save_graph_def=True):
        logging.info("Create CheckpointSaverHook.")
        if saver is not None:
            raise ValueError("You cannot provide both saver and scaffold.")
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        self._save_graph_def = save_graph_def

    def begin(self):
        self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError(
                    "Global step should be created to use CheckpointSaverHook.")

    def after_create_session(self, session, coord):
        global_step = session.run(self._global_step_tensor)
        if self._save_graph_def:
            training_util.write_graph(
                    ops.get_default_graph().as_graph_def(add_shapes=True),
                    self._checkpoint_dir, "graph.pbtxt")
        saver_def = self._get_saver().saver_def if self._get_saver() else None
        graph = ops.get_default_graph()
        meta_graph_def = meta_graph.create_meta_graph_def(
                graph_def=graph.as_graph_def(add_shapes=True), saver_def=saver_def)
        self._summary_writer.add_graph(graph)
        self._summary_writer.add_meta_graph(meta_graph_def)
        # The checkpoint saved here is the state at step "global_step".
        # self._save(session, global_step)

    def after_run(self, run_context, run_values):
        global_step = run_context.session.run(self._global_step_tensor)
        start_time = time.time()
        self._save(run_context.session, global_step)
        save_duration = time.time() - start_time
        logging.info("saving took %d seconds", save_duration)

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        self._save(session, last_step)

    def _save(self, session, step):
        logging.info("Saving checkpoints for %d into %s.", step, self._save_path)

        self._get_saver().save(session, self._save_path, global_step=step,
                                                     write_meta_graph=self._save_graph_def)
        self._summary_writer.add_session_log(
                SessionLog(
                        status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
                step)

    def _get_saver(self):
        if self._saver is not None:
            return self._saver

        # Get saver from the SAVERS collection if present.
        collection_key = ops.GraphKeys.SAVERS
        savers = ops.get_collection(collection_key)
        if not savers:
            raise RuntimeError(
                    "No items in collection {}. Please add a saver to the collection "
                    "or provide a saver or scaffold.".format(collection_key))
        elif len(savers) > 1:
            raise RuntimeError(
                    "More than one item in collection {}. "
                    "Please indicate which one to use by passing it to the constructor."
                    .format(collection_key))

        self._saver = savers[0]
        return savers[0]


class AdaSGDHook():
    def __init__(self, change_step):
        super(AdaSGDHook, self).__init__()
        self._changed_yet = False
        self._change_step = change_step
    
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


class EarlyStoppingHook():
    def __init__(self, threshold):
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


class PrintHook():
    def begin(self):
        self._variables = tf.global_variables()

    def after_run(self, run_context, run_values):
        for var in self._variables:
            print(var.name)


class ScalingFactorHook():
    def __init__(self, batch_size):
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
        

class LossDeltaHook():
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


class ElasticHook():
    def __init__(self, schedule, max_step):
        self._schedule = schedule
        self._max_step = max_step
        self._need_sync = True

    def _build_resize_op(self, config, init_step):
        step = counter(init_step)
        new_size = step_based_schedule(config, step)
        ckpt_tensor = tf.as_string(step + 1)
        resize_op = resize_cluster(ckpt_tensor, new_size)
        return resize_op

    def begin(self):
        self._kungfu_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._advance = tf.assign_add(self._kungfu_step, 1)
        self._sync_op = BroadcastGlobalVariablesOp()
        ckpt = _get_init_step()
        self._init_kungfu_step = tf.assign(self._kungfu_step, int(ckpt))
        self._resize_op = self._build_resize_op(self._schedule, int(ckpt))
        self._reset_global_step = tf.assign(tf.train.get_global_step(),
                                            int(ckpt))

    def after_create_session(self, sess, coord):
        sess.run(self._init_kungfu_step)
        sess.run(self._reset_global_step)

    def before_run(self, run_context):
        kungfu_step = run_context.session.run(self._kungfu_step)
        if kungfu_step >= self._max_step:
            print('request_stop before kungfu_step: %d' % (kungfu_step))
            # run_context.request_stop()
            # FIXME: force quit

        if self._need_sync:
            run_context.session.run(self._sync_op)
            self._need_sync = False

    def after_run(self, run_context, run_values):
        kungfu_step = run_context.session.run(self._kungfu_step)
        changed, keep = run_context.session.run(self._resize_op)
        if changed:
            print('changed on %d' % (kungfu_step))
            self._need_sync = True
            if not keep:
                run_context.request_stop()
                return changed

        kungfu_step = run_context.session.run(self._advance)
        if kungfu_step >= self._max_step:
            print('request_stop on kungfu_step: %d' % (kungfu_step))
            run_context.request_stop()
        return changed

    def end(self, sess):
        global_step = sess.run(tf.train.get_global_step())
        kungfu_step = sess.run(self._kungfu_step)
        print('stopped at global_step: %d, kungfu_step: %d' %
              (global_step, kungfu_step))


class SpotnikHook(tf.train.SessionRunHook):
    def __init__(self, checkpoint_dir):
        schedule = "1:200,2:200,3:200,4:200"
        max_step = 800
        self._elastic_hook = ElasticHook(schedule, max_step)
        self._checkpoint_hook = CheckpointSaverHook(checkpoint_dir)

    def after_create_session(self, session, coord):
        self._elastic_hook.after_create_session(session, coord)
        self._checkpoint_hook.after_create_session(session, coord)

    def after_run(self, run_context, run_values):
        if self._elastic_hook.after_run(run_context, run_values):
            self._checkpoint_hook.after_run(run_context, run_values)

    def before_run(self, run_context):
        self._elastic_hook.before_run(run_context)

    def begin(self):
        self._elastic_hook.begin()
        self._checkpoint_hook.begin()

    def end(self, session):
        self._elastic_hook.end(session)
        self._checkpoint_hook.end(session)
