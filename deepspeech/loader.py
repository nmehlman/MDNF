"""
Automatic speech recognition model
Model contributed by: MITRE Corporation
"""

from typing import Optional

from deepspeech.pytorch_deepspeech import PyTorchDeepSpeech

from armory.utils.external_repo import ExternalRepoImport

# Test for external repo at import time to fail fast
with ExternalRepoImport(
    repo="SeanNaren/deepspeech.pytorch@V3.0",
    experiment="librispeech_asr_snr_undefended.json",
):
    from deepspeech_pytorch.model import DeepSpeech  # noqa: F401


def get_deepspeech_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchDeepSpeech:

    return PyTorchDeepSpeech(**wrapper_kwargs)