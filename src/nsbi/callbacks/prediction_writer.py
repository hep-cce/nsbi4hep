"""Stream per-event scores to disk during the ``predict`` stage.

One file per predict dataloader (i.e. per input file), written batch by batch. Rows come out in
input order -- the predict dataloaders are sequential and unshuffled -- so score row ``k`` is input
row ``k`` and the two join by row number.

The on-disk format is selected by ``predict.format`` and handled by a backend registered in
``FORMATS``. Only ``"csv"`` exists today; a new format is a class with the four-method interface of
``_CsvBackend`` (``extension``/``open``/``write``/``close``) plus a ``FORMATS`` entry -- nothing in
``ScoreWriter`` itself needs to change.
"""

from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from loguru import logger as log


class _CsvBackend:
    """Plain-text CSV: a header line, then one ``float_format`` row per event.

    Streams with ``np.savetxt`` so each batch is appended and released; nothing is buffered beyond
    the batch being written.
    """

    extension = ".csv"

    def __init__(self, float_format: str = "%.9g"):
        # %.9g round-trips float32 exactly; the default %.8g of np.savetxt does not.
        self.float_format = float_format
        self._handle = None

    def open(self, path: Path, columns: list[str]) -> None:
        self._handle = open(path, "w")
        self._handle.write(",".join(columns) + "\n")

    def write(self, rows: np.ndarray) -> None:
        np.savetxt(self._handle, rows, delimiter=",", fmt=self.float_format)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


# Output formats ``predict.format`` can name. Values are zero-argument factories returning a fresh
# backend instance -- one per output file, since a backend owns that file's handle.
FORMATS = {
    "csv": _CsvBackend,
}


class ScoreWriter(BasePredictionWriter):
    """Writes each model's sigmoid output plus the event weight, one row per event.

    Only what a model actually computed is written; nothing is derived.
    Accepts either shape of ``predict_step`` output:

    - a plain tensor of sigmoid outputs (``CARL.predict_step``) -> a ``score`` column;
    - a dict with ``members``, an ``(M, B)`` stack (``MemberEnsemble.predict_step``) -> one
      ``score_i`` column per member, the same quantity ``score`` holds for a single model.

    Args:
        out_dir: Directory the score files are written to (created if absent). ``entry_cli``
            anchors a relative ``predict.out_dir`` to ``datamodule.data_dir`` before passing it
            here, so what arrives is already resolved.
        format: Key into ``FORMATS`` naming the on-disk format. Currently only ``"csv"``.
        suffix: Appended to each input file's stem to name its output file, before the format's
            own extension.
    """

    def __init__(
        self,
        out_dir: str = "predictions",
        format: str = "csv",
        suffix: str = "_scores",
    ):
        super().__init__(write_interval="batch")

        if format not in FORMATS:
            raise ValueError(
                f"Unknown predict format {format!r}; supported: {sorted(FORMATS)}. "
                "Add a backend to nsbi.callbacks.prediction_writer.FORMATS to support another."
            )
        self.out_dir = Path(out_dir)
        self.format = format
        self.suffix = suffix

        # dataloader index -> its open backend / output path / rows written so far
        self._backends: dict[int, object] = {}
        self._paths: dict[int, Path] = {}
        self._rows: dict[int, int] = {}

    def on_predict_start(self, trainer, pl_module) -> None:
        """Resolve one output path per predict dataloader before any batch is scored."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._backends = {}
        self._paths = {}
        self._rows = {}

        extension = FORMATS[self.format].extension
        dataloaders = trainer.predict_dataloaders
        if dataloaders is None:
            return
        if not isinstance(dataloaders, (list, tuple)):
            dataloaders = [dataloaders]

        used: set[Path] = set()
        for i, dataloader in enumerate(dataloaders):
            # PredictDataset records the file it was built from; anything else gets a positional
            # name rather than failing the run.
            source = getattr(getattr(dataloader, "dataset", None), "path", None)
            stem = Path(source).stem if source else f"dataloader_{i}"
            path = self.out_dir / f"{stem}{self.suffix}{extension}"
            # Two inputs can share a stem (same filename in different directories); disambiguate
            # with the dataloader index rather than silently writing one file twice.
            if path in used:
                path = self.out_dir / f"{stem}_{i}{self.suffix}{extension}"
            used.add(path)
            self._paths[i] = path

        log.info(
            "Writing per-event scores ({}) to: {}",
            self.format,
            ", ".join(str(p) for p in self._paths.values()) or "(no predict dataloaders)",
        )

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx=0
    ) -> None:
        """Append one batch of scores (and their event weights) to this dataloader's file."""
        columns = self._columns(prediction)
        weight = batch[1] if isinstance(batch, (tuple, list)) and len(batch) > 1 else None
        if weight is not None:
            columns.append(("weight", weight))

        # Opened on the first batch rather than up front: the column set depends on what the model
        # and dataset actually yield (weights present, how many members), which is only known here.
        backend = self._backends.get(dataloader_idx)
        if backend is None:
            extension = FORMATS[self.format].extension
            fallback = self.out_dir / f"dataloader_{dataloader_idx}{self.suffix}{extension}"
            path = self._paths.setdefault(dataloader_idx, fallback)
            backend = FORMATS[self.format]()
            backend.open(path, [name for name, _ in columns])
            self._backends[dataloader_idx] = backend
            self._rows[dataloader_idx] = 0

        rows = np.column_stack([self._to_numpy(value) for _, value in columns])
        backend.write(rows)
        self._rows[dataloader_idx] += len(rows)

    def on_predict_end(self, trainer, pl_module) -> None:
        """Close every output file and report what was written."""
        for idx, backend in self._backends.items():
            backend.close()
            log.info("Wrote {} rows to {}", self._rows[idx], self._paths[idx])
        self._backends = {}

    @staticmethod
    def _columns(prediction) -> list[tuple[str, torch.Tensor]]:
        """Turn a ``predict_step`` output into named score columns, in order."""
        if isinstance(prediction, dict):
            members = prediction.get("members")
            if members is None:
                raise ValueError(
                    "predict_step returned a dict without a 'members' key; expected an (M, B) "
                    "stack of per-member outputs."
                )
            # Row i is member i, so the column index is the member index that indexes weights.pkl.
            return [(f"score_{i}", members[i]) for i in range(members.shape[0])]
        # Shapes are normalized once, in _to_numpy, for every column alike.
        return [("score", prediction)]

    @staticmethod
    def _to_numpy(value):
        # Widened rather than narrowed: promoting float32 is lossless, so a column_stack of mixed
        # dtypes can never silently truncate a column on the way to the file.
        return value.detach().flatten().double().cpu().numpy()
