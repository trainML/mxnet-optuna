import optuna

study = optuna.load_study(
    study_name="mxnet_pascal_voc_1", storage=os.environ.get("DB_CONNECTION_STRING")
)
print(f"Best Parameters: {study.best_params}")
print(f"Best Score: {study.best_value}")
print(f"Best Trial Details: {study.best_trial}")

df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
print("All Trials")
print("=======================")
print(df)