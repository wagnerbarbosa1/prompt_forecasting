"""
Pipeline de validacao expanding window para previsao de producao de agua
utilizando o modelo TimesFM 2.5 (200M, PyTorch) com covariaveis dinamicas.
"""

import os
import datetime
from pathlib import Path
import logging
 
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from promptcast_forecast.validation.cross_validation import ExpandingWindow, SlidingWindow

from promptcast_forecast.validation.metrics import deterministic_metrics

from promptcast_forecast.visualization.plots import linear_graph
 
import timesfm


def data_cval():
    df = pd.read_csv("/home/wagner-barbosa/Documents/promptcast_forecast/data/processed/guaratuba_processed_1day.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df.isna().any().any():
        raise ValueError("O DataFrame contém valores nulos.")
    #precisa passar por verificação de timestamp tambem --------------------
    return df

df = data_cval()

expanding_window = ExpandingWindow(
    n_samples=len(df),
    trainw= 385, #o tamanho da janela inicial
    step=1, #passo
    horizon=2 #horizonte de previsao
)

torch.set_float32_matmul_precision("high") #melhora o desempenho de multiplicação de matriz em troca de perda de precisão

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

PATCH_LEN = 32
HORIZON_PATCH = 128

def round_up(value, multiple): #função para fazer multiplos de multiple com base no value
    return ((value + multiple - 1) // multiple) * multiple #// sempre arredonda para baixo

resultsEW_val = dict(timestamp=[],  ytrue0=[], ytrue1=[], yhat0=[], yhat1=[])
print("Start Expanding Window")

for i, (trainidxs, testidxs) in enumerate(expanding_window.split()):
    
    context_df = df[trainidxs]

    context_water_produced = context_df["water_produced"].to_numpy()

    current_context_len = len(context_water_produced)
    model.compile(
        timesfm.ForecastConfig(
            max_context=round_up(current_context_len, PATCH_LEN),
            max_horizon=round_up(expanding_window.horizon, HORIZON_PATCH),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=True
        )
    )

    # Covariáveis cobrindo todo o contexto (crescente) + horizonte
    y_t = df[testidxs]
    full_idxs = slice(trainidxs.start, testidxs.stop)
    full_df = df[full_idxs] 

    vol_trend_df = full_df
    vol_trend_df = []
    
    # Executando o modelo
    # Precisa arrumar cada dynamical covariates para cada dataset ----------------------------------------
    point_forecast, quantile_forecast = model.forecast_with_covariates(
        inputs=[context_water_produced],

        dynamic_numerical_covariates={
            #"precipitation_median" : [full_df["precipitation_median"].to_numpy()],

            #"temperature_2m_previsao_mean" : [full_df["temperature_2m_previsao_mean"].to_numpy()],

            #"shortwave_radiation_previsao_std" : [full_df["shortwave_radiation_previsao_std"].to_numpy()],

            #"volume_trend_7d" : [full_df["volume_trend_7d"].to_numpy()], #verificar como lidar com esse volume_trend_7d

            "day" : [full_df["day"].to_numpy()],

            #"dayofweek_sin" : [full_df["dayofweek_sin"].to_numpy()],

            #"dayofweek_cos" : [full_df["dayofweek_cos"].to_numpy()],

            #"season_sin" : [full_df["season_sin"].to_numpy()],

            #"season_cos" : [full_df["season_cos"].to_numpy()]
        },
        dynamic_categorical_covariates={

            #"is_holiday" : [full_df["is_holiday"].astype(str).tolist()]

        },
        xreg_mode="xreg + timesfm"
    )

    median = point_forecast[0]
    ytrue = y_t["water_produced"]
    
    resultsEW_val["timestamp"].append(y_t["timestamp"].iloc[0])

    resultsEW_val["ytrue0"].append(ytrue.iloc[0]) 
    resultsEW_val["ytrue1"].append(ytrue.iloc[1])
    resultsEW_val["yhat0"].append(median[0])
    resultsEW_val["yhat1"].append(median[1])

    print(f"Iteração Window: {i}")


df_result = pd.DataFrame(resultsEW_val)

#Tornar essa lógica genérica ----------------------------------------------------

# Metrics

metricas_dia1 = deterministic_metrics(df_result["ytrue0"], df_result["yhat0"])

metricas_dia2 = deterministic_metrics(df_result["ytrue1"],df_result["yhat1"])

print(metricas_dia1)
print(metricas_dia2)

#----------------------------------------------------

# Output directories

output_models = "output_models"
output_chronos = os.path.join(output_models, "timesfm_chronos")

output_graphs = os.path.join(output_chronos,"timesfm_graph")

output_metrics = os.path.join(output_chronos,"timesfm_metrics_directory")

output_forecast = os.path.join(output_chronos,"timesfm_forecast")

# Creating directories

os.makedirs(output_graphs, exist_ok=True)
os.makedirs(output_metrics, exist_ok=True)
os.makedirs(output_forecast, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d")

# Graphs

graphs_file = os.path.join(output_graphs,f"graph_timesfm_{timestamp}")

linear_graph(df_result,
            highlight_col = "yhat0",
            timestamp_col = "timestamp",
            output_type="output_models",
            model_name="Timesfm2.5")

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
