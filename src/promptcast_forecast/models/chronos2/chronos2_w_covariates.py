import pandas as pd
from pathlib import Path
from chronos import Chronos2Pipeline, ChronosPipeline
import numpy as np
import torch
import argparse
import os
import datetime
from promptcast_forecast.visualization.plots import linear_graph
from promptcast_forecast.validation.metrics import deterministic_metrics
from promptcast_forecast.validation.cross_validation import ExpandingWindow, SlidingWindow


parser = argparse.ArgumentParser(
    description="Chronos forecast for water demand produced\n --guaratuba (guaratuba dataset parameters) " \
    "\n --joinville (joinville dataset parameters)"
)

parser.add_argument(
    "--guaratuba",
    action="store_true",
    help="utiliza o dataset de guaratuba"
)

args = parser.parse_args()

if args.guaratuba:
    df = pd.read_csv("/home/wagner_barbosa/Documents/prompt_forecasting/data/processed/guaratuba_processed_1day.csv") #melhorar isso
else:
    df = pd.read_csv("/home/wagner_barbosa/Documents/prompt_forecasting/data/processed/joinville_processed_1day.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamp is not sorted.")

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map= "cuda" if torch.cuda.is_available() else "cpu" ,
    dtype=torch.float32 #é um tipo bom para IA porque representa grandes escalas sem usar tanta memoria quanto float32
)


# Implementar a sliding windows tambem ------------------------------------------
if args.guaratuba:
    expanding_window = ExpandingWindow(
            n_samples=len(df),
            trainw= 730, #train window
            step=7, #step
            horizon=7 #forecast.
    )
else:    
    expanding_window = ExpandingWindow(
        n_samples=len(df),
        trainw= 385, #train window
        step=7, #step
        horizon=7 #forecast.
    )

resultsEW_val = dict(timestamp=[], ytrue0=[], ytrue1=[], yhat0=[], yhat1=[])
print("Start Expanding Window")

for i, (trainidxs, testidxs) in enumerate(expanding_window.split()):
    # Dados de treino
    context_df = df[trainidxs] 

    # Dados de validaçãok
    y_t = df[testidxs]
    context_future = y_t.drop(["water_produced"], axis=1) 

    timestamp = df[testidxs]

    predictions = pipeline.predict_df(
        context_df,
        id_column="unique_id", #identificador de série temporal, se igual p/ todas as linhas então todas as linhas pertencem a mesma serie temporal
        future_df=context_future, #covariates futuras
        timestamp_column="timestamp",
        batch_size=1, #o tamanho do batch são quantas séries ele vai prever.
        quantile_levels=[0.1, 0.5, 0.9],
        prediction_length=expanding_window.horizon,
        target="water_produced"
)

    median = np.array(predictions["0.5"])

    ytrue = y_t["water_produced"]

    for k in range(expanding_window.horizon):
        resultsEW_val["ytrue"].append(ytrue.iloc[k]) #append adiciona os valores em blocos (tanto faz se é string, macaco, banana)
        resultsEW_val["yhat"].append(median[k])
        resultsEW_val["timestamp"].append(y_t["timestamp"].iloc[k])

    print(i)

df_result = pd.DataFrame(resultsEW_val)

metricas = deterministic_metrics(df_result["ytrue0"], df_result["yhat0"])

print(metricas)


# Output directories

output_models = "output_models"
output_chronos = os.path.join(output_models, "output_chronos")

output_graphs = os.path.join(output_chronos,"chronos_graph")

output_metrics = os.path.join(output_chronos,"chronos_metrics_directory")

output_forecast = os.path.join(output_chronos,"chronos_forecast")

# Creating directories

os.makedirs(output_graphs, exist_ok=True)
os.makedirs(output_metrics, exist_ok=True)
os.makedirs(output_forecast, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d")

# Graphs

graphs_file = os.path.join(output_graphs,f"graph_chronos_{timestamp}")

linear_graph(df_result,
            highlight_col = "yhat0",
            timestamp_col = "timestamp",
            output_type="output_models",
            model_name="Chronos2")

# Metrics

metrics_file = os.path.join(output_metrics,f"metrics_chronos_{timestamp}.csv")

metrics_df = pd.DataFrame({
    "model": ["CHRONOS-EW"],

    "rmse_day1": [metricas["rmse"]],
    "mae_day1": [metricas["mae"]],
    "mape_day1": [metricas["mape"]],
    "r2_day1": [metricas["r2"]],

    "split": ["test"],

    "step": [expanding_window.step],
    "horizon": [expanding_window.horizon],
    "trainw_start": [expanding_window.trainw],
    "n_samples": [expanding_window.n_samples],
})

metrics_df.to_csv(
    metrics_file,
    index=False
)

# Saving file

values_file = os.path.join(output_forecast,f"results_chronos_{timestamp}.csv")

df_result.to_csv(values_file, index=False)
