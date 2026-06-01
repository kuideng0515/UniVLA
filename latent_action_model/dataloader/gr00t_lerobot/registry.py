"""
Centralized registry that auto-discovers benchmark-specific data configs from
``examples/*/train_files/data_registry/`` and merges them with the base
registries defined in this package.

Three registries are maintained:

* ``DATASET_NAMED_MIXTURES``       – mixture_name → [(dataset, weight, robot_type)]
* ``ROBOT_TYPE_CONFIG_MAP``        – robot_type → DataConfig instance
* ``ROBOT_TYPE_TO_EMBODIMENT_TAG`` – robot_type → EmbodimentTag

``ROBOT_TYPE_TO_EMBODIMENT_TAG`` is **derived** from ``ROBOT_TYPE_CONFIG_MAP``
by reading each DataConfig class's ``embodiment_tag`` classvar (Proposal A).
Classes without the classvar fall back to ``EmbodimentTag.NEW_EMBODIMENT``.
Legacy bench files exposing their own ``ROBOT_TYPE_TO_EMBODIMENT_TAG`` dict are
still honored as overrides for backward compatibility.


Usage::

    from dataloader.gr00t_lerobot.registry import (
        ROBOT_TYPE_CONFIG_MAP,
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
        DATASET_NAMED_MIXTURES,
    )
"""

from __future__ import annotations

import logging

# Base registries (kept as fallback / seed values)
from dataloader.gr00t_lerobot.data_config import (
    ROBOT_TYPE_CONFIG_MAP as _BASE_CONFIG_MAP,
)
from dataloader.gr00t_lerobot.embodiment_tags import (
    EmbodimentTag,  # re-export for convenience
)
from dataloader.gr00t_lerobot.mixtures import (
    DATASET_NAMED_MIXTURES as _BASE_MIXTURES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mutable copies – will be extended by discovered modules
# ---------------------------------------------------------------------------
ROBOT_TYPE_CONFIG_MAP: dict = dict(_BASE_CONFIG_MAP)
DATASET_NAMED_MIXTURES: dict = dict(_BASE_MIXTURES)

# Legacy explicit overrides (rarely needed; prefer the classvar on DataConfig).
_LEGACY_TAG_OVERRIDES: dict = {}


def _derive_tag_map() -> dict:
    """Build robot_type -> EmbodimentTag from ROBOT_TYPE_CONFIG_MAP.

    Reads each DataConfig instance's ``embodiment_tag`` classvar. Falls back to
    ``EmbodimentTag.NEW_EMBODIMENT`` if the classvar is missing. Legacy
    overrides from bench files take precedence.
    """
    out: dict = {}
    for rt, cfg in ROBOT_TYPE_CONFIG_MAP.items():
        tag = getattr(cfg, "embodiment_tag", None)
        if tag is None:
            logger.warning(
                "[registry] DataConfig for robot_type=%r has no `embodiment_tag` "
                "classvar; defaulting to EmbodimentTag.NEW_EMBODIMENT.", rt
            )
            tag = EmbodimentTag.NEW_EMBODIMENT
        out[rt] = tag
    out.update(_LEGACY_TAG_OVERRIDES)
    return out


ROBOT_TYPE_TO_EMBODIMENT_TAG: dict = _derive_tag_map()
