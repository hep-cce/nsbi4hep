"""Evaluate every trained ensemble member over the same events, as one LightningModule.

Wrapping the members in a ``LightningModule`` lets the ``predict`` stage score an ensemble through
exactly the same trainer/dataloader/writer path as a single CARL model.
"""

import lightning as L
import torch
from torch import nn


class MemberEnsemble(L.LightningModule):
    """Runs every member over a batch and returns their sigmoid outputs, shape ``(M, B)``.

    Row ``i`` is member ``i``'s output, ordered by member index, so it lines up with ``w[i]`` from
    ``weights.pkl`` whenever something downstream chooses to combine them. ``ScoreWriter`` writes
    each row as a ``score_i`` column -- the same quantity a single model contributes as ``score``.

    Args:
        members: Trained member models, ordered by member index.
    """

    def __init__(self, members):
        super().__init__()

        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-member sigmoid outputs ``s_i(x)``, shape ``(M, B)``."""
        return torch.stack([member(x).flatten() for member in self.members], dim=0)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Return the per-member outputs under the ``members`` key for ``ScoreWriter``."""
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return {"members": self(x)}
