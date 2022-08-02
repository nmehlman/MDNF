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
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm
import pdb
import torchaudio
from math import log10
import pickle
import os

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch, Preprocessor
import torch
from mdnf.mel2wav.interface import MelVocoder
import sys

from os.path import join

logger = logging.getLogger(__name__)

LOCAL_WEIGHTS_DIR = globals()["__file__"].replace("MDNF_preprocessor.py", "weights") # Need for imports of weights/smoothing files

class MDNF_Torch(PreprocessorPyTorch):
    
    """
    Implements the MDNF defense.
    """

    def __init__(
        self,
        mg_weights_file = 'melGAN.pt',
        nf_level = 0,
        nf_curve_file = 'smoothing_curve',
        apply_pca = False,
        pca_comps = 80,
        apply_fit: bool = False,
        apply_predict: bool = True,
        channels_first: bool = True,
        verbose: bool = False
    ) -> None:

        """Create an instance of the MDNF preprocessor
        :param mg_weights_file: name of file containing melGAN model weights. 
        Should be located in 'weights' directory of the submission repo.
        :param nf_level: noise-flooding amplitude.
        :param nf_curve_file: name of file containing 'informed'smoothing curve. 
        Should be located in 'weights' directory of the submission repo.
        :param apply_pca: flag to apply PCA projection to noise-flooded mel vector.
        :param pca_comps: number of PCA components to use (between 1 and 80).
        :param apply_fit: apply fitting to preprocessor
        :param apply_predict: apply predict to preprocessor
        :param channels_first: for backwards compatibility (unused)
        :param verbose: for backwards compatibility (unused)
        """

        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        
        cuda_idx = torch.cuda.current_device()                                       
        self._device = torch.device("cuda:{}".format(cuda_idx)) 

        ## PCA config
        self.apply_pca = apply_pca
        self.pca_comps = pca_comps

        ## NF config
        self.nf_level = nf_level
        self.informed_smoothing = bool(nf_curve_file)
        if nf_curve_file:
            nf_curve_path = os.path.join(LOCAL_WEIGHTS_DIR, nf_curve_file)
            try:
                self.smoothing_curve = pickle.load(open(nf_curve_path, 'rb')).to(self.device)
            except FileNotFoundError:
                print("Smoothing curve file not found. Ensure that the file is located in the 'mdnf/weights' directory")
                sys.exit(1)

        ## MelGAN setup
        mg_weights_path = os.path.join(LOCAL_WEIGHTS_DIR, mg_weights_file)
        try:
            self.mel_GAN = MelVocoder(path=mg_weights_path, device=self.device)
        except FileNotFoundError:
            print("MelGAN weights file not found. Ensure that the file is located in the 'mdnf/weights' directory")
            sys.exit(1)

    def forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:

        x = torchaudio.functional.resample(x, 16000, 22000)

        ndim = x.ndim # Check shape
        if ndim == 1:
            x = x.unsqueeze(0)
        
        mels = self.mel_GAN(x) # Log mel spectrogram
        
        if self.apply_pca: # Compute PCA projection matrix

            m_mels = torch.mean( mels, 2, keepdim=True ) # Centering
            mels = mels - m_mels

            U, _, _ = torch.svd(mels, compute_uv=True)
            U = U[ :, :self.pca_comps ]

            Proj_Mtx = torch.matmul( torch.transpose(U,1,2),U ) # PCA projection matrix

        if self.nf_level:
            
            n = torch.randn(mels.size()).to(self.device) # Noise matrix
            
            if self.informed_smoothing: # Shaping
                n = n * self.smoothing_curve
            
            # Fame-based normalization
            noise_frame_energy = torch.norm(n, 2, dim=1).squeeze() # Compute energy of each frame
            mels_frame_energy = torch.norm(mels, 2, dim=1).squeeze()
            n = n * mels_frame_energy/noise_frame_energy # Normalize
            
            mels = mels + self.nf_level * n

        if self.apply_pca: # Project onto PCA subspace
            mels = torch.matmul( Proj_Mtx, mels)
            mels = mels + m_mels # Add back mean

        x = self.mel_GAN.inverse(mels) # Inversion

        x = torchaudio.functional.resample(x, 22000, 16000)
        
        return x, y