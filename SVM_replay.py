import os
import pathlib

from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.adapters.pyod_adapters import OneClassSVMAdapter
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.strategies.replay.buffers.adaptive_balanced import (
    AdaptiveBalancedReplayBuffer
)
from pyclad.strategies.replay.selection.random import RandomSelection
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.data.readers.concepts_readers import read_dataset_from_npy
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario

os.chdir('/net/afscra/people/plgayahorava')

dataset = read_dataset_from_npy(
        pathlib.Path("/net/afscra/people/plgayahorava/physionet.org/files/chbmit/1.0.0/concepts_freq_ch_mean.npy"), dataset_name="EEG_CHBMIT_Concepts"
    )

model = OneClassSVMAdapter(max_iter = 5000,cache_size = 500, tol = 1e-1, nu = 0.2)
replay_buffer = AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=10000)
strategy = ReplayEnhancedStrategy(model,replay_buffer)

time_callback = TimeEvaluationCallback()
memory_callback = MemoryUsageCallback()
metric_callback = ConceptMetricCallback(
    base_metric=RocAuc(), metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()]
)

# Execute the concept agnostic scenario
scenario = ConceptIncrementalScenario(dataset=dataset, strategy=strategy, callbacks=[metric_callback, time_callback,memory_callback])
scenario.run()

# Save the results
output_writer = JsonOutputWriter(pathlib.Path("output_svm_replay_incremental.json"))
output_writer.write([model, dataset, strategy, metric_callback, time_callback,memory_callback])

