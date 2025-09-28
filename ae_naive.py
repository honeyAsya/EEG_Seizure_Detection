import os
import logging
import pathlib

from torch import nn
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.models.autoencoder.autoencoder import Autoencoder
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.data.readers.concepts_readers import read_dataset_from_npy
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario


os.chdir('/net/afscra/people/plgayahorava')


dataset = read_dataset_from_npy(
        pathlib.Path("/net/afscra/people/plgayahorava/physionet.org/files/chbmit/1.0.0/concepts_freq_ch_mean.npy"), dataset_name="EEG_CHBMIT_Concepts"
    )

logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()])

input_channels = dataset._train_concepts[0].data.shape[1]

encoder = nn.Sequential(
    nn.Linear(input_channels, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 2)  
)

decoder = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, input_channels),
    nn.Sigmoid(),
)

model = Autoencoder(encoder, decoder)

strategy = NaiveStrategy(model)
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
output_writer = JsonOutputWriter(pathlib.Path("output_ae_naive_incremental.json"))
output_writer.write([model, dataset, strategy, *callbacks])