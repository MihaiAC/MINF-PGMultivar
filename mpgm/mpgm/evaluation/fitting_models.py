import time

from mpgm.mpgm.model_fitters.prox_grad_fitters import *
from mpgm.mpgm.evaluation.generating_samples import SampleParamsWrapper, SampleParamsSave
from sqlitedict import SqliteDict
from typing import Optional
from mpgm.mpgm.evaluation.preprocessing import *

from mpgm.mpgm.models.TPGM import TPGM
from mpgm.mpgm.models.Model import Model

class FitParamsSave():
    def __init__(self):
        self.random_seed = None
        self.samples_id = None
        self.samples_file_name = None
        self.theta_init = None

        self.theta_fit = None
        self.likelihoods = None
        self.converged = None
        self.conditions = None
        self.regularization_paths = None
        self.avg_node_fit_time = None

        self.model = (None, None)
        self.fitter = (None, None)
        self.fit_time = None

        self.preprocessor = None

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
        self.FPS.theta_fit = None

        self._model = None
        self._fitter = None

    @property
    def preprocessor(self):
        return self.FPS.preprocessor

    @preprocessor.setter
    def preprocessor(self, value:Preprocessor):
        self.FPS.preprocessor = value

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

    def fit_model_and_save(self,
                           fit_id:str,
                           fit_file_name:str,
                           parallelize:Optional[bool]=True,
                           samples_file_name:Optional[str]=None,
                           samples_id:Optional[str]=None,
                           theta_init:Optional[np.ndarray]=None):
        if samples_file_name is not None:
            self.FPS.samples_file_name = samples_file_name

        if samples_id is not None:
            self.FPS.samples_id = samples_id

        SPS = SampleParamsWrapper.load_samples(self.FPS.samples_id, self.FPS.samples_file_name)
        samples = SPS.samples

        if theta_init is None and self.FPS.theta_init is None:
            nr_variables = samples.shape[1]
            self.FPS.theta_init = np.random.normal(0, 0.001, (nr_variables, nr_variables))
        else:
            # samples array should have dims (nr_samples, nr_variables);
            # params array should have dims (nr_variables, nr_variables);
            assert samples.shape[1] == theta_init.shape[0], \
                   "Initial dimensions for theta_init do not match with the dimensions of the selected samples"
            self.FPS.theta_init = theta_init

        if self.FPS.preprocessor is not None:
            self.FPS.preprocessor.preprocess(samples)

        fit_start_time = time.time()
        fit_results = self.fitter.call_fit_node(nll=self.model.calculate_nll,
                                                grad_nll=self.model.calculate_grad_nll,
                                                data_points=samples,
                                                theta_init=self.FPS.theta_init,
                                                parallelize=parallelize)
        self.FPS.fit_time = time.time() - fit_start_time

        self.model.theta = fit_results[0]
        self.model = self.model # Update FPS' model.

        self.FPS.theta_fit = fit_results[0]
        self.FPS.likelihoods = fit_results[1]
        self.FPS.converged = fit_results[2]
        self.FPS.conditions = fit_results[3]
        self.FPS.regularization_paths = fit_results[4]
        self.FPS.avg_node_fit_time = fit_results[5]

        fits_dict = SqliteDict('./' + fit_file_name, autocommit=True)
        fits_dict[fit_id] = self.FPS
        fits_dict.commit()
        fits_dict.close()

        return self.FPS.theta_fit

    def fit_lpgm_and_save(self,
                         fit_ids:List[str],
                         fit_file_name:str,
                         samples_file_name:Optional[str]=None,
                         samples_id:Optional[str]=None,
                         theta_init:Optional[np.ndarray]=None,):
        if samples_file_name is not None:
            self.FPS.samples_file_name = samples_file_name

        if samples_id is not None:
            self.FPS.samples_id = samples_id

        SPS = SampleParamsWrapper.load_samples(self.FPS.samples_id, self.FPS.samples_file_name)
        samples = SPS.samples

        if theta_init is None and self.FPS.theta_init is None:
            assert 0 == 1, "Provided an empty theta_init -> exit."
        else:
            # samples array should have dims (nr_samples, nr_variables);
            # params array should have dims (nr_variables, nr_variables);
            assert samples.shape[1] == theta_init.shape[0], \
                   "Initial dimensions for theta_init do not match with the dimensions of the selected samples"
            self.FPS.theta_init = theta_init

        if self.FPS.preprocessor is not None:
            self.FPS.preprocessor.preprocess(samples)

        fit_start_time = time.time()
        fit_results = self.fitter.call_fit_node(nll=self.model.calculate_nll,
                                                grad_nll=self.model.calculate_grad_nll,
                                                data_points=samples,
                                                theta_init=self.FPS.theta_init)
        self.FPS.fit_time = time.time() - fit_start_time

        self.FPS.likelihoods = []
        self.FPS.converged = []
        self.FPS.conditions = []
        self.FPS.regularization_paths = []
        self.FPS.avg_node_fit_time = 0

        ahats_main = fit_results[0]
        for ii in range(len(fit_ids)):
            theta_fit = ahats_main[ii, :, :]

            self.model.theta = theta_fit
            self.model = self.model

            self.FPS.theta_fit = theta_fit

            fits_dict = SqliteDict('./' + fit_file_name, autocommit=True)
            fits_dict[fit_ids[ii]] = self.FPS
            fits_dict.commit()
            fits_dict.close()

        return np.array([])


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
    FPW.fitter = Prox_Grad_Fitter(alpha=0.065, early_stop_criterion='weight')
    FPW.preprocessor = ClampMax(10)
    theta_final = FPW.fit_model_and_save(fit_id=fit_id,
                                        fit_file_name=fit_file_name,
                                        parallelize=True)

    # FPS = FitParamsWrapper.load_fit(fit_id, fit_file_name)
    # print(theta_fit)