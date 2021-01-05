import optuna
import os

study = optuna.create_study(
    study_name="mxnet_pascal_voc_1", storage=os.environ.get("DB_CONNECTION_STRING")
)
