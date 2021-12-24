from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow_model_optimization as tfmot

pruning_params = {
    "pruning_schedule": sparsity.PolynomialDecay(
        initial_sparsity=0.85,
        final_sparsity=0.95,
        begin_step=2000,
        end_step=5000,
        frequency=10,
    )
}

pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=10
    )
}
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
