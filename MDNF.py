# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This module implements the MDNF defence as a PreprocessorPyTorch subclass.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from audio_export_wrapper import audio_export
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm
import pdb
import torchaudio
from math import log10
import pickle

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch, Preprocessor
import torch
from mel2wav import MelVocoder

from os.path import join

logger = logging.getLogger(__name__)

class modelArgs:
    def __init__(self, path):
        self.model_path = path

class MDNF_Torch(PreprocessorPyTorch):
    
    """
    Implements the melGAN defense.
    """

    def __init__(
        self,
        nf_level = 0,
        nf_curve_file = None,
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:

        """Create an instance of the perceptual denoiser."""

        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        
        cuda_idx = torch.cuda.current_device()                                       
        self._device = torch.device("cuda:{}".format(cuda_idx)) 

        self.nf_level = nf_level
        self.informed_smoothing = bool(nf_curve_file)
        if nf_curve_file:
            self.smoothing_curve = pickle.load(open(nf_curve_file, 'rb')).to(self.device)

        self.resample1 = torchaudio.transforms.Resample(16000, 22000)
        self.resample2 = torchaudio.transforms.Resample(22000, 16000)

        self.mel_GAN = MelVocoder(path = "", github=True, model_name="multi_speaker")

    def forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:

        x = self.resample1(x)
        mels = self.mel_GAN(x)
        if self.nf_level:
            n = torch.randn(mels.size()).to(self.device)
            if self.informed_smoothing:
                n = n * self.smoothing_curve
            n = n/torch.norm(n)  # Normalize by energy
            mels = mels + self.nf_level*n

        x = self.mel_GAN.inverse(mels)
        x = self.resample2(x)
        
        return x, y