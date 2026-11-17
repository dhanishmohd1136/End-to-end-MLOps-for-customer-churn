from mlflow import MlflowClient
from pathlib import  Path
import mlflow


client = MlflowClient()


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT /"data"/"data.csv"
MODEL_NAME = "ChurnModel"


def rollback(version):
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias='champion',
        version=version
    )

    print("Champion model is changed")



if __name__ == "__main__":
    rollback(2)