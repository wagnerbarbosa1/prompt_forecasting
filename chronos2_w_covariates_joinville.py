import pandas as pd
from chronos import Chronos2Pipeline, ChronosPipeline
import numpy as np
import torch
import os
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt



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

class SlidingWindow:
    def __init__(self, n_samples, trainw, step, horizon):
        self.n_samples = n_samples
        self.trainw = trainw
        self.step = step
        self.horizon = horizon

    def split(self):
        for k in range (self.trainw, self.n_samples - self.horizon + 1, self.step):
            trainidxs = slice(k - self.trainw, k)
            testidxs = slice(k, k + self.horizon)

            yield trainidxs, testidxs
    
df = pd.read_csv("/home/wagner-barbosa/Documents/promptcast_forecast/data/processed/guaratuba_processed_1daysin.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
    torch_dtype=torch.float32 #é um tipo bom para IA porque representa grandes escalas sem usar tanta memoria quanto float32
)

expanding_window_test = ExpandingWindow(
    n_samples=len(df),
    trainw= 385, #o tamanho da janela inicial
    step=7, #passo
    horizon=7 #horizonte de previsao
)

resultsEW_val = dict(timestamp=[], ytrue0=[], ytrue1=[], yhat0=[], yhat1=[])
print("Start Expanding Window")

for i, (trainidxs, testidxs) in enumerate(expanding_window_test.split()):
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
        freq="D",
        timestamp_column="timestamp",
        batch_size=1, #o tamanho do batch são quantas séries ele vai prever.
        quantile_levels=[0.1, 0.5, 0.9],
        prediction_length=expanding_window_test.horizon,
        target="water_produced"
)

    median = np.array(predictions["0.5"])

    ytrue = y_t["water_produced"]
    
    resultsEW_val["ytrue0"].append(ytrue.iloc[0]) #append adiciona os valores em blocos (tanto faz se é string, macaco, banana)
    resultsEW_val["ytrue1"].append(ytrue.iloc[1])
    resultsEW_val["yhat0"].append(median[0])
    resultsEW_val["yhat1"].append(median[1])
    resultsEW_val["timestamp"].append(y_t["timestamp"].iloc[0])

    print(i)


df_result = pd.DataFrame(resultsEW_val)
print("NaNs por coluna:\n", df_result[["ytrue0", "yhat0", "ytrue1", "yhat1"]].isna().sum())
#df_result["timestamp"] = df["timestamp"].iloc[expanding_window_test.trainw:] #aplicar logs para verificação de dimensionalidade do dataset
#df_result["timestamp"] = pd.to_datetime(df_result["timestamp"])

rmse0 = round(np.sqrt(mean_squared_error(df_result["ytrue0"], df_result["yhat0"])), 4)
mae0 = round(mean_absolute_error(df_result["ytrue0"], df_result["yhat0"]), 4)
mape0 = round(mean_absolute_percentage_error(df_result["ytrue0"], df_result["yhat0"]), 4)
r20 = round(r2_score(df_result["ytrue0"], df_result["yhat0"]), 4)

rmse1 = round(np.sqrt(mean_squared_error(df_result["ytrue1"].dropna(), df_result["yhat1"].dropna())), 4)
mae1 = round(mean_absolute_error(df_result["ytrue1"].dropna(), df_result["yhat1"].dropna()), 4)
mape1 = round(mean_absolute_percentage_error(df_result["ytrue1"].dropna(), df_result["yhat1"].dropna()), 4)
r21 = round(r2_score(df_result["ytrue1"].dropna(), df_result["yhat1"].dropna()), 4)

print("RMSE : ", rmse0)
print("R2 : ", r20)
print("MAE : ", mae0)
print("MAPE : ", mape0)

print("\nRMSE : ", rmse1)
print("R2 : ", r21)
print("MAE : ", mae1)
print("MAPE : ", mape1)

fig, ax = plt.subplots(figsize=(12, 6))
plt.style.use('default')

ax.plot(df_result["timestamp"], df_result["ytrue0"], color='#d3d3d3', alpha = 0.8, linewidth=1.0, label='Water_produced')

ax.plot(df_result["timestamp"], df_result["yhat0"], color="#2554b9", linewidth=1.3, alpha=0.6, label='Chronos2')

plt.title('Chronos2 x Water_produced', 
          fontsize=16, loc='left', pad=20, color='#333333')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')


ax.tick_params(axis='both', colors='#666666')


plt.legend(frameon=False, loc='upper left', fontsize=9)


plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

output_graph = "chronos_graph"
if not os.path.exists(output_graph):
    os.makedirs(output_graph)


current_time = datetime.datetime.now()

timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")

graphs_file = os.path.join(output_graph, f"graph_chronos_{timestamp}")

plt.savefig(graphs_file)

# diretórios
output_dir = "chronos_metrics_directory"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

metrics_file = os.path.join(output_dir, f"metrics_chronos_{timestamp}.csv")

with open(metrics_file, "w") as f: #with garante que eu abra e feche o arquivo
    f.write("model,rmse0,mae0,mape0,r20,rmse1,mae1,mape1,r21,split\n")

# Salva as metricas
with open(metrics_file, "a") as f:
    f.write(f"CHRONOS-EW,{rmse0},{mae0},{mape0},{r20},{rmse1},{mae1},{mape1},{r21},test\n")

# CSV com ytrue e yhat
output_dir2 = "chronos_forecast_yhat_ytrue"
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

values_file = os.path.join(output_dir2, f"results_chronos_{timestamp}.csv")

df_result.to_csv(values_file, index=False)
