import os
import time
from tensorflow.python.training import session_run_hook
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training import training_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import meta_graph
from tensorflow.core.util.event_pb2 import SessionLog


class SpotnikCheckpointSaverHook(session_run_hook.SessionRunHook):
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
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
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
    if global_step % 10 == 0:
      start_time = time.time()
      self._save(run_context.session, global_step)
      save_duration = time.time() - start_time
      print("saving took ", save_duration, " seconds")


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
