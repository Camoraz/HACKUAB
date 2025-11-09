import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from tbats import TBATS
import matplotlib.pyplot as plt

def main():
    month_map = {
        'GENER': 1, 'FEBRER': 2, 'MARÇ': 3, 'ABRIL': 4, 'MAIG': 5, 'JUNY': 6,
        'JULIOL': 7, 'AGOST': 8, 'SETEMBRE': 9, 'OCTUBRE': 10, 'NOVEMBRE': 11, 'DESEMBRE': 12
    }

    files = {
        2021: "timeseries/2021.csv",
        2022: "timeseries/2022.csv",
        2023: "timeseries/2023.csv",
    }

    dfs = []
    for year, path in files.items():
        df = pd.read_csv(path)
        df_long = df.melt(id_vars=['NOM ESTACIÓ'], var_name='MES', value_name='DEMANDA')
        df_long['YEAR'] = year
        df_long['MONTH_NUM'] = df_long['MES'].map(month_map)
        df_long['DATE'] = pd.to_datetime(dict(year=df_long['YEAR'], month=df_long['MONTH_NUM'], day=1))
        dfs.append(df_long)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all[['NOM ESTACIÓ', 'DATE', 'DEMANDA']].dropna()
    df_all['DEMANDA'] = pd.to_numeric(df_all['DEMANDA'], errors='coerce')

    station_name = "HOSPITAL DE BELLVITGE"
    series = (
        df_all[df_all["NOM ESTACIÓ"] == station_name]
        .sort_values("DATE")
        .set_index("DATE")["DEMANDA"]
    )

    estimator = TBATS(seasonal_periods=[12], use_arma_errors=True, n_jobs=1)
    model = estimator.fit(series)
    forecast_steps = 12
    forecast = model.forecast(steps=forecast_steps)

    forecast_index = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(),
                                   periods=forecast_steps, freq='MS')

    # 2024 データを読み込む
    df_2024 = pd.read_csv("timeseries/2024.csv")
    df_2024_long = df_2024.melt(id_vars=['NOM ESTACIÓ'], var_name='MES', value_name='DEMANDA')
    df_2024_long['MONTH_NUM'] = df_2024_long['MES'].map(month_map)
    df_2024_long['DATE'] = pd.to_datetime(dict(year=2024, month=df_2024_long['MONTH_NUM'], day=1))
    df_2024_series = (
        df_2024_long[df_2024_long["NOM ESTACIÓ"] == station_name]
        .sort_values("DATE")
        .set_index("DATE")["DEMANDA"]
    )

    plt.figure(figsize=(12,6))
    plt.plot(series.index, series.values, label="Actual 2021-2023", marker='o')
    plt.plot(forecast_index, forecast, label="Forecast 2024", marker='x')
    plt.plot(df_2024_series.index, df_2024_series.values, label="Actual 2024", marker='s')
    plt.title(f"TBATS Forecast vs Actual for {station_name}")
    plt.xlabel("Date")
    plt.ylabel("Monthly Demand")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_file = "timeseries.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Gráfico guardado en {output_file}")


if __name__ == "__main__":
    main()

