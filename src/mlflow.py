import hashlib
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
from src.train import train_model

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT /"data"/"data.csv"
MODEL_NAME = "ChurnModel"


mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
mlflow.set_experiment("ChurnModelExperiment")

def get_dataset_version(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dataset_version = get_dataset_version(DATA_PATH)

    with mlflow.start_run():

        model,X_train,metrics = train_model(DATA_PATH)

        mlflow.log_params({
            "model":"logistic regression",
            "random_state":43
        })

        mlflow.log_metrics(metrics)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format='cloudpickle',
            input_example=X_train.head(3)
        )

        register = mlflow.register_model(
            model_uri=model_info.model_uri,
            name= MODEL_NAME
        )

        if metrics['f1'] >.70:
            client = MlflowClient()

            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="champion",
                version=register.version
            )

            print("\n Model is Promoted\n")
        
        else:

            print('\n Model failed to Promote \n')

        
        print('\n Metrics \n',metrics)
        print('\n Model version : ',register.version)
        print('\n Dataset version : ',dataset_version)



if __name__ == "__main__":
    main()