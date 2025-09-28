import os
import logging
import pathlib

from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.pyod_adapters import IsolationForestAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.strategies.replay.buffers.adaptive_balanced import (
    AdaptiveBalancedReplayBuffer
)
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.selection.random import RandomSelection
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.data.readers.concepts_readers import read_dataset_from_npy
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario


os.chdir('/net/afscra/people/plgayahorava')


logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

    
dataset = read_dataset_from_npy(
        pathlib.Path("/net/afscra/people/plgayahorava/physionet.org/files/chbmit/1.0.0/concepts_freq_ch_mean.npy"), dataset_name="EEG_CHBMIT_Concepts"
    )

model = IsolationForestAdapter()
replay_buffer = AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=10000)
strategy = ReplayEnhancedStrategy(model,replay_buffer)

callbacks = [
    ConceptMetricCallback(
        base_metric=RocAuc(),
        metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
    ),
    TimeEvaluationCallback(), 
    MemoryUsageCallback()
]

scenario = ConceptIncrementalScenario(dataset, strategy=strategy, callbacks=callbacks)
scenario.run()

# Save the results
output_writer = JsonOutputWriter(pathlib.Path("output_is_replay_incremental.json"))
output_writer.write([model, dataset, strategy, *callbacks])