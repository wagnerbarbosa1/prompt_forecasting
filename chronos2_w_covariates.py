import pandas as pd
from chronos import Chronos2Pipeline, ChronosPipeline
import numpy as np
import torch
import os
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from siaqua.visualization.plots import linear_graph
from siaqua.evaluation.metrics import deterministic_metrics
from siaqua.evaluation.cross_validation import ExpandingWindow, SlidingWindow

    
df = pd.read_csv("/home/wagner-barbosa/Documents/IC/siaqua/data/processed/joinville_processed_1day.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
    torch_dtype=torch.float32 #é um tipo bom para IA porque representa grandes escalas sem usar tanta memoria quanto float32
)

expanding_window_test = ExpandingWindow(
    n_samples=len(df),
    trainw= 385, #train window
    step=1, #step
    horizon=2 #forecast
)

resultsEW_val = dict(timestamp=[], ytrue0=[], ytrue1=[], yhat0=[], yhat1=[])
print("Start Expanding Window")

for i, (trainidxs, testidxs) in enumerate(expanding_window_test.split()):
    # Dados de treino
    context_df = df[trainidxs] 

    # Dados de validaçãok
    y_t = df[testidxs]
    context_future = y_t.drop(["water_produced", "volume_trend_7d"], axis=1) 

    timestamp = df[testidxs]

    predictions = pipeline.predict_df(
        context_df,
        id_column="unique_id", #identificador de série temporal, se igual p/ todas as linhas então todas as linhas pertencem a mesma serie temporal
        future_df=context_future, #covariates futuras
        timestamp_column="timestamp",
        batch_size=1, #o tamanho do batch são quantas séries ele vai prever.
        quantile_levels=[0.1, 0.5, 0.9],
        prediction_length=expanding_window_test.horizon,
        target="water_produced"
)

    median = np.array(predictions["0.5"])

    #Refazer essa lógica para se tornar genérica. ------------------------------------

    ytrue = y_t["water_produced"]
    
    resultsEW_val["ytrue0"].append(ytrue.iloc[0]) #append adiciona os valores em blocos (tanto faz se é string, macaco, banana)
    resultsEW_val["ytrue1"].append(ytrue.iloc[1])
    resultsEW_val["yhat0"].append(median[0])
    resultsEW_val["yhat1"].append(median[1])
    resultsEW_val["timestamp"].append(y_t["timestamp"].iloc[0])

    # -----------------------------------------------------------------------------------

    print(i)

df_result = pd.DataFrame(resultsEW_val)

# Metrics

metricas_dia1 = deterministic_metrics(df_result["ytrue0"], df_result["yhat0"])

metricas_dia2 = deterministic_metrics(df_result["ytrue1"],df_result["yhat1"])

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

linear_graph(df_result,"Chronos2","timestamp", graphs_file)

# Metrics

metrics_file = os.path.join(output_metrics,f"metrics_chronos_{timestamp}.csv")

metrics_df = pd.DataFrame({
    "model": ["CHRONOS-EW"],

    "rmse_day1": [metricas_dia1["rmse"]],
    "mae_day1": [metricas_dia1["mae"]],
    "mape_day1": [metricas_dia1["mape"]],
    "r2_day1": [metricas_dia1["r2"]],

    "rmse_day2": [metricas_dia2["rmse"]],
    "mae_day2": [metricas_dia2["mae"]],
    "mape_day2": [metricas_dia2["mape"]],
    "r2_day2": [metricas_dia2["r2"]],

    "split": ["test"],

    "step": [expanding_window_test.step],
    "horizon": [expanding_window_test.horizon],
    "trainw_start": [expanding_window_test.trainw],
    "n_samples": [expanding_window_test.n_samples],
})

metrics_df.to_csv(
    metrics_file,
    index=False
)

# Saving file

values_file = os.path.join(output_forecast,f"results_chronos_{timestamp}.csv")

df_result.to_csv(values_file, index=False)
