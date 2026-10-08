"""Plot all seven forecast leads from completed text-dataset Qwen evaluations."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.pipelines.qwen_text_experiment import OUTPUT, RUNS, summarize


def main():
    summarize()
    destination = OUTPUT / 'plots'
    destination.mkdir(exist_ok=True)
    metrics = pd.read_csv(OUTPUT / 'metrics.csv')
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    for city in ('joinville', 'guaratuba'):
        for slug in ('1_7b', '4b'):
            label = 'Qwen3 1.7B' if slug == '1_7b' else 'Qwen3 4B'
            fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, constrained_layout=True)
            for ax, days in zip(axes, (7, 15)):
                frame = pd.read_csv(OUTPUT / f'{slug}_{days}' / city / 'qwen/predictions.csv', parse_dates=['timestamp'])
                score = metrics.query('dataset == @city and slug == @slug and context == @days').iloc[0]
                ax.plot(frame.timestamp, frame.y_true, color='#818181', linewidth=1.3, label='Real')
                ax.plot(frame.timestamp, frame.y_pred, color='#2478b5', linewidth=1.2, label='Previsto')
                ax.set_title(f'Contexto: {days} dias | RMSE: {score.rmse:,.2f} | R²: {score.r2:.3f}', loc='left')
                ax.set_ylabel('Produção de água (m³)')
                ax.grid(alpha=.2)
                ax.legend(loc='upper right', ncol=2)
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
            fig.suptitle(f'{city.title()} — {label} | Dataset textual e prompts em linguagem natural', fontsize=18)
            fig.savefig(destination / f'{city}_{slug}.png', dpi=150)
            plt.close(fig)
        score = metrics.query('dataset == @city').copy()
        labels = [f'{row.model}\n{row.context} dias' for row in score.itertuples()]
        fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
        bars = ax.bar(labels, score.rmse, color=['#2478b5', '#75b5df', '#db8438', '#f3bd89'])
        ax.bar_label(bars, labels=[f'{value:,.2f}' for value in score.rmse], padding=6, fontsize=15)
        ax.set_ylim(0, score.rmse.max()*1.2)
        ax.set_ylabel('RMSE (m³) — menor é melhor')
        ax.set_title(f'{city.title()} — Qwen | Comparação dos contextos fixos', fontsize=21)
        ax.grid(axis='y', alpha=.2)
        fig.savefig(destination / f'{city}_rmse.png', dpi=150)
        plt.close(fig)
    print(f'Saved six figures in {destination}')


if __name__ == '__main__':
    main()
