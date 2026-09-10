"""Model construction from trained checkpoints.

Written for the ``predict`` stage -- driven by ``main_function`` in ``entry_cli``, which this
module only supplies the thing that does the scoring -- but ``build_model_from_checkpoint`` is
equally the right entry point for every path that loads trained weights into a fresh model:
``finetune`` and ``test`` in ``entry_cli``, and the member loop in ``ensemble_fit``. A single
checkpoint needs no assembly beyond that, so the rest of what lives here is the ensemble path:
loading the members that ``do_ensemble_train`` produced.
"""

import json

import hydra
import torch
from loguru import logger as log
from omegaconf import DictConfig

from nsbi.models.ensemble import MemberEnsemble


# Hyperparameters that do not describe the architecture, so a config/checkpoint disagreement is
# harmless at inference and not worth warning about: the optimizer is never constructed.
_NON_ARCHITECTURAL_HPARAMS = {"learning_rate"}


def build_model_from_checkpoint(cfg: DictConfig, ckpt_path, warn_on_mismatch: bool = True):
    """Instantiate the model the checkpoint was trained as, and load its weights.

    Lightning modules that call ``save_hyperparameters`` (as ``CARL`` does) record the architecture
    they were built with, and those weights only fit that architecture -- so whenever weights are
    loaded back the checkpoint is authoritative and ``cfg.model`` is a guess. Stored hyperparameters
    are applied over the config block, keeping its ``_target_``; any that disagree are warned about,
    and a checkpoint that stored none leaves the config untouched.

    Without this, a checkpoint whose architecture differs from ``cfg.model`` fails in
    ``load_state_dict`` as a wall of tensor shapes that names no config key -- and only after the
    input files have been read.

    Args:
        cfg: Full config; ``cfg.model`` supplies the ``_target_`` and any values the checkpoint
            does not record.
        ckpt_path: Checkpoint to build from and load.
        warn_on_mismatch: Log the disagreements. Ensemble members share an architecture, so their
            loop reports it once rather than once per member.

    Returns:
        The model, with the checkpoint's weights loaded.
    """
    # weights_only=False: the hyperparameters are pickled Python objects, not tensors.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    # Lightning bookkeeping rather than a constructor argument -- it is what makes
    # load_from_checkpoint route construction through jsonargparse.
    hparams.pop("_instantiator", None)

    # Only keys the config already declares are applied: a checkpoint may carry extras the
    # configured _target_ does not accept, and passing those would be a TypeError.
    overrides = {key: value for key, value in hparams.items() if key in cfg.model}

    differing = {
        key: (cfg.model[key], value)
        for key, value in overrides.items()
        if cfg.model[key] != value and key not in _NON_ARCHITECTURAL_HPARAMS
    }
    if differing and warn_on_mismatch:
        for key, (configured, stored) in sorted(differing.items()):
            log.warning(
                "cfg.model.{} is {!r} but {} was trained with {!r}; using the checkpoint's value.",
                key,
                configured,
                ckpt_path,
                stored,
            )

    model = hydra.utils.instantiate(cfg.model, **overrides)
    model.load_state_dict(ckpt["state_dict"])
    return model


def _member_count(ensemble_dir, ensemble_cfg) -> int:
    """How many members to load: the manifest's count, else ``ensemble.size``.

    Unlike training and fitting, where ``size`` is an input, prediction scores an ensemble that
    already exists -- so the manifest on disk is authoritative. Deferring to ``ensemble.size`` here
    would mean a config naming fewer members than were trained silently scores a subset, with no
    error and a short score file. ``ensemble.size`` is still the fallback for ensembles trained
    before manifests existed, where ``load_member_ckpts`` globs the member directories.
    """
    from nsbi.tools.ensemble_fit import MANIFEST_NAME

    manifest_path = ensemble_dir / MANIFEST_NAME
    if not manifest_path.exists():
        size = ensemble_cfg.get("size", None)
        if size is None:
            raise FileNotFoundError(
                f"No manifest at {manifest_path} and no ensemble.size configured, so the number of "
                "members to load is unknown. Set ensemble.size, or re-run training to write a "
                "manifest."
            )
        return int(size)

    with open(manifest_path) as f:
        size = len(json.load(f)["members"])
    configured = ensemble_cfg.get("size", None)
    if configured is not None and int(configured) != size:
        log.warning(
            "ensemble.size is {} but {} lists {} members; using {}.",
            configured,
            manifest_path,
            size,
            size,
        )
    return size


def build_member_ensemble(cfg: DictConfig) -> MemberEnsemble:
    """Load every trained member into a ``MemberEnsemble``.

    Reads the member checkpoints through the same manifest the weight fit uses, so it is agnostic
    to which training path (single-device or distributed) wrote them. It deliberately does not read
    ``weights.pkl``: scoring writes the members' own outputs, so an ensemble can be scored before
    its weights are fit, and a later re-fit does not invalidate the score files.

    Each member is built from its own checkpoint's stored hyperparameters, not from ``cfg.model``.
    Members share an architecture, so a disagreement is reported once rather than once per member.

    Args:
        cfg: Full config; ``cfg.ensemble`` locates the ensemble and ``cfg.model`` supplies the
            ``_target_`` to load each checkpoint into.

    Returns:
        The assembled ensemble, in eval mode.
    """
    from nsbi.tools.ensemble_fit import load_member_ckpts, resolve_ensemble_dir

    ensemble_cfg = cfg.get("ensemble", {})
    ensemble_dir = resolve_ensemble_dir(ensemble_cfg)
    size = _member_count(ensemble_dir, ensemble_cfg)

    members = []
    for i, ckpt in enumerate(load_member_ckpts(ensemble_dir, size)):
        log.info("Member {}: loading {}", i, ckpt)
        member = build_model_from_checkpoint(cfg, ckpt, warn_on_mismatch=(i == 0))
        member.eval()
        members.append(member)

    log.info("Predicting with {} ensemble members from {}", size, ensemble_dir)
    ensemble = MemberEnsemble(members)
    ensemble.eval()
    return ensemble
