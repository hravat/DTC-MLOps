import os
import pickle
import click
import mlflow


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)



mlflow.autolog()
relative_path_db = os.path.join("..", "mlflow-db", "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(relative_path_db)}")

artifact_location = "file:///E:/GitHub/DTC MLOps/mlflow-artifacts/02-Experiment-Tracking"
experiment_name = "02-Experiment-Tracking"

# Check if the experiment already exists
if mlflow.get_experiment_by_name(experiment_name) is None:
    experiment_id = mlflow.create_experiment(experiment_name
                                             ,artifact_location=artifact_location
                                            )
    print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
else:
    print(f"Experiment '{experiment_name}' already exists.")

mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    @click.command()
    @click.option(
        "--data_path",
        default="..//Homework-02-Prepocess-Output",
        help="Location where the processed NYC taxi trip data was saved"
    )
    def run_train(data_path: str):

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)

print(f"Tracking URI:- {mlflow.get_tracking_uri()}")
print(f"Artifact Path:- {artifact_location}")

# Get experiment details
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment:
    print(f"Artifact Location for '{experiment_name}': {experiment.artifact_location}")
else:
    print(f"Experiment '{experiment_name}' not found.")

if __name__ == '__main__':
    run_train()
