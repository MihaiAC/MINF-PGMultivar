from mpgm.mpgm.model_fitters.prox_grad_fitters import Prox_Grad_Fitter

from mpgm.mpgm.evaluation.evaluation_metrics import *


def get_sample_name(experiment_name:str, batch_nr:int, sample_nr:int) -> str:
    sample_name = experiment_name + "_batch_" + str(batch_nr) + "_sample_" + str(sample_nr)
    return sample_name

def get_fit_sample_name(experiment_name:str, batch_nr:int, sample_nr:int) -> str:
    return "fit_" + get_sample_name(experiment_name, batch_nr, sample_nr)

def generate_batch_samples_vary_seed(SPW:SampleParamsWrapper, experiment_name:str, batch_nr:int, samples_per_batch:int,
                                     random_seeds:List[int], samples_file_name:str):
    assert samples_per_batch == len(random_seeds), "Must have the same number of random seeds as the number of samples you " \
                                            "want to generate"
    for sample_nr in range(samples_per_batch):
        SPW.random_seed = random_seeds[sample_nr]
        sample_name = get_sample_name(experiment_name, batch_nr, sample_nr)
        SPW.generate_samples_and_save(sample_name, samples_file_name)
        print("Sampling " + sample_name + " finished.")

def vary_nr_samples_and_generate_samples(SPW:SampleParamsWrapper, experiment_name:str, samples_per_batch:int,
                                         samples_file_name:str, samples_numbers:List[int]):
    nr_batches = len(samples_numbers)
    random_seeds = list(range(samples_per_batch))
    for batch_nr, nr_samples in enumerate(samples_numbers):
        SPW.nr_samples = nr_samples
        generate_batch_samples_vary_seed(SPW, experiment_name, batch_nr, samples_per_batch, random_seeds,
                                         samples_file_name)


def fit_all_batches_all_samples(FPW:FitParamsWrapper, fit_file_name:str, experiment_name:str, samples_per_batch:int,
                                nr_batches:int, theta_init:np.ndarray):
    for batch_nr in range(nr_batches):
        for sample_nr in range(samples_per_batch):
            sample_name = get_sample_name(experiment_name, batch_nr, sample_nr)
            fit_id = get_fit_sample_name(experiment_name, batch_nr, sample_nr)
            FPW.fit_model_and_save(fit_id, fit_file_name, samples_id=sample_name, theta_init=theta_init)



experiment_name = "lattice_same_neg_weight_vary_nr_samples"
samples_file_name = "samples.sqlite"
# SGW = SampleParamsWrapper(nr_variables=20, nr_samples=10, random_seed=0, sample_init=np.zeros((20, )))
#
# SGW.graph_generator = LatticeGraphGenerator(sparsity_level=0)
# SGW.weight_assigner = Bimodal_Distr_Weight_Assigner(neg_mean=-0.1, threshold=1, std=0)
# SGW.model = TPGM(R=10)
# SGW.sampler = TPGMGibbsSampler(burn_in = 200,
#                                thinning_nr = 50)

# samples_per_batch = 5
# samples_numbers = [10, 30, 100, 300, 600]
# vary_nr_samples_and_generate_samples(SGW, experiment_name, samples_per_batch, samples_file_name, samples_numbers)


FPW = FitParamsWrapper(random_seed=2,
                       samples_file_name=samples_file_name)

FPW.model = TPGM(R=10)
FPW.fitter = Prox_Grad_Fitter(alpha=0.3, early_stop_criterion='likelihood')
FPW.fit_model_and_save(fit_id=fit_id, fit_file_name=fit_file_name, parallelize=False)

FPS = FitParamsWrapper.load_fit(fit_id, fit_file_name)