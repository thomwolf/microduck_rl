"""Regression tests for W&B configuration in HF Jobs submissions."""

from mjlab_microduck.hf_jobs import _wandb_job_env


def test_no_wandb_disables_logging_in_job_environment() -> None:
    """--no-wandb must prevent the remote process from requesting a login."""
    assert _wandb_job_env(disabled=True) == {"WANDB_MODE": "disabled"}


def test_wandb_enabled_does_not_override_mode() -> None:
    """Authenticated W&B jobs should retain the library's normal mode."""
    assert _wandb_job_env(disabled=False) == {}
