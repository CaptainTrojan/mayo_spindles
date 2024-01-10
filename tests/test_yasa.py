from mayo_spindles.yasa_util import yasa_predict
from mayo_spindles.dataloader import SpindleDataModule


def test_yasa_predict():
    datamodule = SpindleDataModule('data', 30, intracranial_only=False, batch_size=1)
    datamodule.setup()
    val_dataloader = datamodule.val_dataloader()
    signals, metadata = next(iter(val_dataloader))
    metadata = metadata[0]
    sf = datamodule.dataset._common_sampling_rate
    rel_pow = 0.1
    corr = 0.5
    rms = 1.3
    overlap_thresh = 10

    y_pred = yasa_predict(signals, metadata, sf, rel_pow, corr, rms, overlap_thresh)
    a = 4