"""Dataset-specific readers, each registered with ``@register_dataset`` (Phase 1).

Importing this package registers every loader. A ``@register_dataset`` decorator only
runs when its module is imported, so without this a config naming ``bigvul`` would fail
with "unknown dataset" despite the loader existing — the classic registry-discovery
trap. Adding a loader means adding one import here.
"""

from vulnloc.data.loaders.bigvul import BigVulLoader

__all__ = ["BigVulLoader"]
