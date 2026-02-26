from pathlib import Path

from loguru import logger


def find_latest_checkpoint(ckpt_path: str | Path, templates=None):
    if templates is None:
        templates = ["*.ckpt"]
    elif isinstance(templates, str):
        templates = [templates]
    elif not isinstance(templates, list):
        raise ValueError("Templates should be a string or a list of strings.")

    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        logger.warning("Checkpoint path does not exist: {}", ckpt_path)
        return None

    ckpt_files: list[Path] = []
    for template in templates:
        ckpt_files.extend(list(ckpt_path.rglob(template)))

    return max(ckpt_files, key=lambda p: p.stat().st_ctime) if ckpt_files else None
