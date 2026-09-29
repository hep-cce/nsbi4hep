# TODO

- **Add CI.** Run unit tests and other validation tests.
- **Make a generic script launcher.** `scripts/` currently holds examples for Polaris/PBS. Implement
  a generic launcher that is also configurable for site specifics.
- **Read input files in chunks.** The datamodule loads whole files into memory: a loader returns
  `(X, w)` as arrays and `PredictDataset` / `BalancedDataset` index them, so a file that does not
  fit in memory cannot be scored or trained on. Support a chunked path (e.g. an `IterableDataset`
  fed by a chunked reader), which also means revising what a loader is required to return.
- **Migrate ALICE/TAYLR off direct imports of the examples package.** `datasets/jointlikelihood.py`
  (`AliceDataModule`) and `datasets/coefficient.py` import `nsbi.examples.physics` at module scope
  (`analysis.zz4l`, `hstar.c6` / `hstar.eft`, `simulation.mcfm` / `simulation.msq`). The
  `BalancedDataModule` + CARL path reaches the same package only through a config `_target_`; these
  should do the same, so the example domain code stays a replaceable template rather than a
  framework dependency.
- **Test support for non-NVIDIA GPUs.** Test running for ex. Intel GPUs (Aurora) which may require
  generalizing the accelerator choice and NVIDIA-specific code in the framework.
