if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


import mlflow 
import os
import pickle 

def get_dir_size(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.pkl'):
                full_path = os.path.join(root, f)
                size = os.path.getsize(full_path)
                print(f"Found .pkl file: {full_path}, Size: {size} bytes")
                return size
    return total

@data_exporter
def export_data(dict_param, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
        
    """
    EXPERIMENT_NAME = "register-model-from-mage"
    model=dict_param['model']
    vec = dict_param['vectorizer']

    print(f"Model Intercept: {model.intercept_}")
    print(f"MLflow version: {mlflow.__version__}")

    mlflow.set_tracking_uri("http://mlflow_server:5000")

    artifact_location='/mlflow-artifacts'
    
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME
                                                 ,artifact_location=artifact_location
                                                )
        print(f"Experiment '{EXPERIMENT_NAME}' created with ID: {experiment_id}")
    else:
        print(f"Experiment '{EXPERIMENT_NAME}' already exists.")

    print(f"Tracking URI: {mlflow.get_tracking_uri()}")    


    # Specify your data exporting logic here

    mlflow.set_experiment(EXPERIMENT_NAME)
    
    vec_path = '/tmp/dict_vectorizer.pkl'
    with open(vec_path, 'wb') as f:
        pickle.dump(vec, f)
    

    with mlflow.start_run():
        mv=mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="model-from-mage"
        )

        
        mlflow.log_artifact(vec_path, 
                            artifact_path="dict_vectorizer")

    print(f"Model URL :- {mv.model_uri}")

    local_path = mlflow.artifacts.download_artifacts(mv.model_uri)

    # Calculate size
    size_bytes = get_dir_size(local_path)
    size_mb = size_bytes / (1024 ** 2)

    print(f"Model size for '{mv.model_uri}': {size_bytes} Bytes")
