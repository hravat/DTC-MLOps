import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


relative_path_db = os.path.join("..", "mlflow-db", "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(relative_path_db)}")

artifact_location = "file:///E:/GitHub/DTC MLOps/mlflow-artifacts/random-forest-hyperopt"
experiment_name = "random-forest-hyperopt"

# Check if the experiment already exists
if mlflow.get_experiment_by_name(experiment_name) is None:
    experiment_id = mlflow.create_experiment(experiment_name
                                             ,artifact_location=artifact_location
                                            )
    print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
else:
    print(f"Experiment '{experiment_name}' already exists.")

mlflow.set_experiment(experiment_name)

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
     default="..//Homework-02-Prepocess-Output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):
    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
    def objective(params):
        with mlflow.start_run(nested=True):
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_params(params)
            mlflow.log_metrics({"rmse": rmse})
            return {'loss': rmse, 'status': STATUS_OK}
    
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }
    
    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
        )

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment:
    print(f"Artifact Location for '{experiment_name}': {experiment.artifact_location}")
else:
    print(f"Experiment '{experiment_name}' not found.")

if __name__ == '__main__':
    run_optimization()

     