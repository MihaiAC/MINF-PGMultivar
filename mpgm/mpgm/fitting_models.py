from mpgm.mpgm.model_fitters.prox_grad_fitters import *
from mpgm.mpgm.generating_samples import SampleParamsSave, SampleParamsWrapper
from sqlitedict import SqliteDict
from typing import Optional

from mpgm.mpgm.models.TPGM import TPGM
from mpgm.mpgm.models.Model import Model

class FitParamsSave():
    def __init__(self):
        self.random_seed = None
        self.samples_id = None
        self.samples_file_name = None
        self.theta_init = None

        self.theta_final = None
        self.likelihoods = None
        self.converged = None
        self.conditions = None

        self.model = (None, None)
        self.fitter = (None, None)

    @property
    def model_name(self):
        return self.model[0]

    @property
    def model_params(self):
        return self.model[1]

    @property
    def fitter_name(self):
        return self.fitter[0]

    @property
    def fitter_params(self):
        return self.fitter[1]

class FitParamsWrapper():
    def __init__(self, random_seed:int, samples_file_name:Optional[str]=None, samples_id:Optional[str]=None,
                 theta_init:Optional[np.ndarray]=None):
        self.FPS = FitParamsSave()
        self.FPS.random_seed = random_seed
        self.FPS.samples_id = samples_id
        self.FPS.samples_file_name = samples_file_name
        self.FPS.theta_init = theta_init
        self.FPS.theta_final = None

        self._model = None
        self._fitter = None

    @property
    def random_seed(self):
        return self.FPS.random_seed

    @random_seed.setter
    def random_seed(self, value:int):
        self.FPS.random_seed = value

    @property
    def samples_file_name(self):
        return self.FPS.samples_file_name

    @samples_file_name.setter
    def samples_file_name(self, value:str):
        self.FPS.samples_file_name = value

    @property
    def samples_id(self):
        return self.FPS.samples_id

    @samples_id.setter
    def samples_id(self, value:str):
        self.FPS.samples_id = value

    @property
    def theta_init(self):
        return self.FPS.theta_init

    @theta_init.setter
    def theta_init(self, value:np.ndarray):
        self.FPS.theta_init = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value:Model):
        self._model = value
        self.FPS.model = (type(value).__name__, vars(value))

    @property
    def fitter(self):
        return self._fitter

    @fitter.setter
    def fitter(self, value:Prox_Grad_Fitter):
        self._fitter = value
        self.FPS.fitter = (type(value).__name__, vars(value))

    def fit_model_and_save(self, fit_id:str, fit_file_name:str, parallelize:Optional[bool]=True,
                           samples_file_name:Optional[str]=None, samples_id:Optional[str]=None,
                           theta_init:Optional[np.ndarray]=None):
        if samples_file_name is not None:
            self.FPS.samples_file_name = samples_file_name

        if samples_id is not None:
            self.FPS.samples_id = samples_id

        SPS = SampleParamsWrapper.load_samples(self.FPS.samples_id, self.FPS.samples_file_name)
        samples = SPS.samples

        if theta_init is None or self.FPS.theta_init is None:
            nr_variables = samples.shape[1]
            self.FPS.theta_init = np.random.normal(-0.03, 0.001, (nr_variables, nr_variables))
            print(self.FPS.theta_init)
        else:
            assert samples.shape[1] == theta_init.shape[0] and samples.shape[1] == theta_init.shape[1], \
                   "Initial dimensions for theta_init do not match with the dimensions of the selected samples"
            self.FPS.theta_init = theta_init

        theta_final, likelihoods, converged, conditions = self.fitter.call_fit_node(nll=self.model.calculate_nll,
                                                                                    grad_nll=self.model.calculate_grad_nll,
                                                                                    data_points=samples,
                                                                                    theta_init=self.FPS.theta_init,
                                                                                    parallelize=parallelize)
        self.model.theta = theta_final
        self.model = self.model # Update FPS' model.

        self.FPS.theta_final = theta_final
        self.FPS.likelihoods = likelihoods
        self.FPS.converged = converged
        self.FPS.conditions = conditions

        fits_dict = SqliteDict('./' + fit_file_name, autocommit=True)
        fits_dict[fit_id] = self.FPS
        fits_dict.close()

        return theta_final

    @staticmethod
    def load_fit(fit_id:str, sqlite_file_name:str) -> FitParamsSave:
        fits_dict = SqliteDict('./' + sqlite_file_name, autocommit=True)
        FPS = fits_dict[fit_id]
        fits_dict.close()
        return FPS

if __name__ == "__main__":
    samples_file_name = "samples.sqlite"
    samples_id = "PleaseWork"

    fit_id = "debugProxGrad"
    fit_file_name = "fit_models.sqlite"

    FPW = FitParamsWrapper(random_seed=2,
                           samples_file_name=samples_file_name,
                           samples_id=samples_id)

    FPW.model = TPGM(R=10)
    FPW.fitter = Prox_Grad_Fitter(alpha=0.3)
    FPW.fit_model_and_save(fit_id=fit_id,
                           fit_file_name=fit_file_name,
                           parallelize=False)

    FPS = FitParamsWrapper.load_fit(fit_id, fit_file_name)
    print(FPS.theta_final)