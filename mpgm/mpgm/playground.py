from mpgm.mpgm.model_fitters.prox_grad_fitters import *
from mpgm.mpgm.generating_samples import SampleParamsSave, SampleParamsWrapper
from mpgm.mpgm.fitting_models import FitParamsSave, FitParamsWrapper

from sqlitedict import SqliteDict
from typing import Optional

from mpgm.mpgm.models import Model
from mpgm.mpgm.models.TPGM import TPGM
from mpgm.mpgm.models.Model import Model

import numpy as np


def soft_thresholding_prox_operator(x, threshold):
    x_plus = x - threshold
    x_minus = x + threshold

    return x_plus * (x_plus > 0) + x_minus * (x_minus < 0)

if __name__ == "__main__":
    samples_file_name = "samples.sqlite"
    samples_id = "PleaseWork"

    fit_id = "debugProxGrad"
    fit_file_name = "fit_models.sqlite"

    SPS = SampleParamsWrapper.load_samples(samples_id, samples_file_name)
    print(SPS.model_params)

    # a = np.array([2, 4, 5, 0.5, -0.95, -1.25])
    # print(soft_thresholding_prox_operator(a, 1))



