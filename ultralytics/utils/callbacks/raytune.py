# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch."""
    # Handle both older (<2.7) and newer (>=2.7) Ray versions where the session accessor changed
    ray_session_mod = ray.train._internal.session
    if hasattr(ray_session_mod, "_get_session"):
        _sess = ray_session_mod._get_session()
    else:
        # Fallback for newer Ray versions that expose get_session()
        _sess = ray_session_mod.get_session()

    if _sess:  # Only report if we're inside an active Ray Tune session
        metrics = trainer.metrics
        metrics["epoch"] = trainer.epoch
        session.report(metrics)


callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)
