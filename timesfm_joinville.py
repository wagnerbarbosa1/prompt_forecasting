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
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
 
import timesfm

logger = logging.getLogger(__name__)

class ExpandingWindow:
    def __init__(self, n_samples, trainw, step, horizon):
        self.n_samples = n_samples
        self.trainw = trainw
        self.step = step
        self.horizon = horizon

    def split(self):
        for k in range(self.trainw, self.n_samples - self.horizon + 1, self.step): #+1 só porque o range exclui o limite superior
            #           k é o conjunto de treinamento
            trainidxs = slice(0, k)
            testidxs = slice(k, k + self.horizon)

            yield trainidxs, testidxs

def data_cval():
    df = pd.read_csv("/home/wagner-barbosa/Documents/promptcast_forecast/data/processed/guaratuba_processed_1daysin.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df.isna().any().any():
        raise ValueError("O DataFrame contém valores nulos.")
    return df

df = data_cval()

expanding_window_test = ExpandingWindow(
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

for i, (trainidxs, testidxs) in enumerate(expanding_window_test.split()):
    
    context_df = df[trainidxs]

    context_water_produced = context_df["water_produced"].to_numpy()

    current_context_len = len(context_water_produced)
    model.compile(
        timesfm.ForecastConfig(
            max_context=round_up(current_context_len, PATCH_LEN),
            max_horizon=round_up(expanding_window_test.horizon, HORIZON_PATCH),
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
    point_forecast, quantile_forecast = model.forecast_with_covariates(
        inputs=[context_water_produced],
        dynamic_numerical_covariates={
            "relative_humidity_mean" : [full_df["relative_humidity_mean"].to_numpy()],
            "temperature_mean" : [full_df["temperature_mean"].to_numpy()],
            "radiation_mean" : [full_df["radiation_mean"].to_numpy()],
            "precipitation_mean" : [full_df["precipitation_mean"].to_numpy()], #verificar como lidar com esse volume_trend_7d
            "day" : [full_df["day"].to_numpy()],
            "dayofweek_sin" : [full_df["dayofweek_sin"].to_numpy()],
            "dayofweek_cos" : [full_df["dayofweek_cos"].to_numpy()],
            "season_sin" : [full_df["season_sin"].to_numpy()],
            "season_cos" : [full_df["season_cos"].to_numpy()]
        },
        dynamic_categorical_covariates={
            "is_holiday_ctba_gtba_jve" : [full_df["is_holiday_ctba_gtba_jve"].astype(str).tolist()],
            "is_weekend" : [full_df["is_weekend"].astype(str).tolist()],
            "is_carnival" : [full_df["is_carnival"].astype(str).tolist()],
            "is_school_recess_pr" : [full_df["is_school_recess_pr"].astype(str).tolist()]
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

print("\nTotal de linhas em df_result:", len(df_result))
print("NaNs por coluna:\n", df_result[["ytrue0", "yhat0", "ytrue1", "yhat1"]].isna().sum())

# Métricas Horizonte 0
rmse0 = round(np.sqrt(mean_squared_error(df_result["ytrue0"], df_result["yhat0"])), 3)
mae0 = round(mean_absolute_error(df_result["ytrue0"], df_result["yhat0"]), 3)
mape0 = round(mean_absolute_percentage_error(df_result["ytrue0"], df_result["yhat0"]), 3)
r20 = round(r2_score(df_result["ytrue0"], df_result["yhat0"]), 3)

# Métricas Horizonte 1
rmse1 = round(np.sqrt(mean_squared_error(df_result["ytrue1"], df_result["yhat1"])), 3)
mae1 = round(mean_absolute_error(df_result["ytrue1"], df_result["yhat1"]), 3)
mape1 = round(mean_absolute_percentage_error(df_result["ytrue1"], df_result["yhat1"]), 3)
r21 = round(r2_score(df_result["ytrue1"], df_result["yhat1"]), 3)

print("RMSE 0 : ", rmse0)
print("R2 0 : ", r20)
print("MAE 0 : ", mae0)
print("MAPE 0 : ", mape0)

print("\nRMSE 1 : ", rmse1)
print("R2 1 : ", r21)
print("MAE 1 : ", mae1)
print("MAPE 1 : ", mape1)

fig, ax = plt.subplots(figsize=(12, 6))
plt.style.use('default')

ax.plot(df_result["timestamp"], df_result["ytrue0"], color='#d3d3d3', alpha = 0.8, linewidth=1.0, label='Water_produced') #colocar Métricas nos gráficos

ax.plot(df_result["timestamp"], df_result["yhat0"], color="#2554b9", linewidth=1.3, alpha=0.6, label='Timesfm')

plt.title('Timesfm x Water_produced', 
          fontsize=16, loc='left', pad=20, color='#333333')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')


ax.tick_params(axis='both', colors='#666666')


plt.legend(frameon=False, loc='upper left', fontsize=9)


plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

output_graph = "timesfm_graph"
if not os.path.exists(output_graph):
    os.makedirs(output_graph)


current_time = datetime.datetime.now()

timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")

graphs_file = os.path.join(output_graph, f"graph_timesfm_{timestamp}")

plt.savefig(graphs_file)

# diretórios
output_dir = "timesfm_metrics_directory"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

metrics_file = os.path.join(output_dir, f"metrics_timesfm_{timestamp}.csv")

with open(metrics_file, "w") as f: #with garante que eu abra e feche o arquivo
    f.write("model,rmse0,mae0,mape0,r20,rmse1,mae1,mape1,r21,split\n")

# Salva as metricas
with open(metrics_file, "a") as f:
    f.write(f"timesfm-EW,{rmse0},{mae0},{mape0},{r20},{rmse1},{mae1},{mape1},{r21},test\n")

# CSV com ytrue e yhat
output_dir2 = "timesfm_forecast_yhat_ytrue"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

values_file = os.path.join(output_dir2, f"results_timesfm_{timestamp}.csv")

df_result.to_csv(values_file, index=False)