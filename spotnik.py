import os
import time
import re

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
from kungfu.tensorflow.ops import consensus
from kungfu.tensorflow.experimental.hook import ElasticHook


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
        size_str = str(size)
        if size_str in self.examples_per_second:
            self.examples_per_second[size_str]["steps"] = self.examples_per_second[size_str]["steps"] + 1
        else:
            self.examples_per_second[size_str] = dict()
            self.examples_per_second[size_str]["steps"] = 1
            self.examples_per_second[size_str]["throughputs"] = []
        self.examples_per_second[size_str]["throughputs"].append((self.batch_size * size) / self.time_difference)
        print("global examples per second ", self.examples_per_second)

    def end(self, session):
        for siz in self.examples_per_second:
            sum = 0
            for ele in self.examples_per_second[siz]["throughputs"]:
                sum = sum + ele
            avg = sum / self.examples_per_second[siz]["steps"]
            print("size ", siz, " has average throughput of ", avg)


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


class ConsensusHook():
    def begin(self):
        self._consensus_op = [consensus(var) for var in tf.global_variables()]

    def after_run(self, run_context, run_values):
        consensus_checks = run_context.session.run(self._consensus_op)
        for check in consensus_checks:
            if not check:
                print("DIFF")


class CheckpointAndRestoreHook():
    def __init__(self):
        self._checkpoints = dict()
        self._checkpoint_placeholers = []
        self._restore_checkpoint = False
        self._branch = 0

        # intermediately
        self._checkpoint_key = (0,0)

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
        self._checkpoints[(global_step, self._branch)] = run_context.session.run(self._global_variables)
        print("checkpoint at ", global_step)
        self._checkpoint_key = (global_step, self._branch)

    def before_run(self, run_context):
        global_step = run_context.session.run(tf.train.get_or_create_global_step())
        print("global_step ", global_step)
        if global_step == 310:
            self._restore_checkpoint = True
        else:
            self._restore_checkpoint = False
        if self._restore_checkpoint:
            run_context.session.run(self._restore_op, feed_dict={ckpt_pl.name: ckpt for ckpt_pl, ckpt in zip(self._checkpoint_placeholers, self._checkpoints[self._checkpoint_key])})
            self._branch = self._branch + 1
            print("restored checkpoint at ", global_step)


class SpotnikHook(tf.train.SessionRunHook):
    def __init__(self, num_train_steps):
        pass

    def after_create_session(self, session, coord):
        pass

    def after_run(self, run_context, run_values):
        pass

    def before_run(self, run_context):
        pass

    def begin(self):
        pass

    def end(self, session):
        pass
