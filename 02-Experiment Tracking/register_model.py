import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

relative_path_db = os.path.join("..", "mlflow-db", "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(relative_path_db)}")

artifact_location_rfh = "file:///E:/GitHub/DTC MLOps/mlflow-artifacts/random-forest-hyperopt"
artifact_location_rfbm = "file:///E:/GitHub/DTC MLOps/mlflow-artifacts/random-forest-best-models"



if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME
                                             ,artifact_location=artifact_location_rfbm
                                            )
    print(f"Experiment '{EXPERIMENT_NAME}' created with ID: {experiment_id}")
else:
    print(f"Experiment '{EXPERIMENT_NAME}' already exists.")


mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        new_params = {}
        for param in RF_PARAMS:
            new_params[param] = int(params[param])

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))
        mlflow.log_metric("test_rmse", test_rmse)


@click.command()
@click.option(
    "--data_path",
    default="..//Homework-02-Prepocess-Output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs( 
        experiment_ids=experiment.experiment_id,
        order_by=["metrics.test_rmse ASC"],
        max_results=1)[0]

    print(f"Best run ID: {best_run.info.run_id}")
    print(f"Lowest Test RMSE: {best_run.data.metrics['test_rmse']}")
    # Register the best model
    model_uri = f"runs:/{best_run.info.run_id}/sklearn-model"
    mv = mlflow.register_model(model_uri, "RandomForestRegressionModel")
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")


if __name__ == '__main__':
    run_register_model()
