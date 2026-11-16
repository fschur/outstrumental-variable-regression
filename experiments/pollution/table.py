"""Reproduce the PM2.5 mortality comparison in five US metro areas."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import torch

from experiments.common.baselines import AnchorRegression, EnvironmentFixedEffects, LinearNeuralRegression, OLS
from ovr import DirectOVR, SmoothedOVR
from .data import metro_days


def weather_adjusted_fe(data, x, y, env, calendar):
    # City intercepts, shared calendar coefficients and city-specific weather curves.
    controls = [np.eye(data.city.nunique())[env], calendar]
    for city in np.unique(env):
        rows = env == city
        for values, df in ((data.temp4, 6), (data.humidity, 3)):
            knots = np.quantile(values[rows], np.linspace(0, 1, df + 1))
            spline = CubicSpline(knots, np.eye(df + 1)[:, 1:], bc_type="natural")
            basis = np.zeros((len(data), df))
            basis[rows] = spline(values[rows])
            controls.append(basis)
    controls = np.column_stack(controls)
    xy = np.column_stack((x, y))
    x_res, y_res = (xy - controls @ np.linalg.lstsq(controls, xy, rcond=None)[0]).T
    return (x_res @ y_res) / (x_res @ x_res)


def fit_methods(data, y):
    cities = pd.Categorical(data.city)
    env = cities.codes.copy()
    ref = cities.categories.get_loc("Los Angeles")
    months = data.date.dt.to_period("M").astype(str).to_numpy()
    season = (data.date.dt.month % 12) // 3
    calendar = pd.get_dummies(pd.DataFrame({"season": season, "weekday": data.date.dt.dayofweek}),
                              columns=["season", "weekday"], drop_first=True, dtype=float).to_numpy()
    x = data[["pm25"]].to_numpy() / 10
    design = np.column_stack((x, calendar))
    slopes = {}
    for name, method in (("SOVR", SmoothedOVR(reference=ref)), ("DOVR", DirectOVR(reference=ref))):
        slopes[name] = method.fit(x, y, env, groups=months, controls=calendar).coef_.item()
    slopes["OLS"] = OLS().fit(design, y, env).coef_[0]
    slopes["FE"] = EnvironmentFixedEffects().fit(design, y, env).coef_[0]
    slopes["FE (weather-adjusted)"] = weather_adjusted_fe(data, x, y, env, calendar)
    slopes["Anchor (γ = 2)"] = AnchorRegression(gamma=2).fit(design, y, env).coef_[0]
    slopes["LNN"] = LinearNeuralRegression().fit(design, y, env, groups=months).coef_[0]

    # City indicators instrument PM2.5. Calendar controls enter both stages.
    instruments = np.column_stack((np.eye(len(cities.categories))[env], calendar))
    projected_x = instruments @ np.linalg.lstsq(instruments, x, rcond=None)[0]
    slopes["Environment IV"] = OLS().fit(np.column_stack((projected_x, calendar)), y).coef_[0]
    for i, city in enumerate(cities.categories):
        rows = env == i
        slopes[f"Per-environment OLS ({city})"] = OLS().fit(design[rows], y[rows]).coef_[0]
    return pd.Series(slopes)


def poisson_reference(data):
    with TemporaryDirectory() as tmp:
        panel = Path(tmp) / "metro_days.csv"
        data.to_csv(panel, index=False)
        command = ["Rscript", "--vanilla", str(Path(__file__).with_name("reference.R")), str(panel)]
        output = subprocess.check_output(command, text=True)
    absolute, percent = map(float, output.split())
    return absolute, percent


def main():
    data = metro_days(Path("data/pollution"))
    rate = (1e6 * data.death / data.population).to_numpy()
    if not np.all(rate > 0):
        raise ValueError("The log-rate comparison requires positive mortality on every eligible day.")
    torch.set_num_threads(1)
    absolute = fit_methods(data, rate)
    percent = 100 * np.expm1(fit_methods(data, np.log(rate)))
    ref_absolute, ref_percent = poisson_reference(data)
    results = pd.DataFrame({"deaths_per_million": absolute, "change_percent": percent,
                            "absolute_error_pp": (percent - ref_percent).abs()})
    reference = pd.DataFrame([[ref_absolute, ref_percent, np.nan]], columns=results.columns,
                             index=["Schwartz-style Poisson reference"])
    table = pd.concat([reference, results]).rename_axis("method")
    output = Path("results/pollution/table.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output)
    print(table.to_string(float_format=lambda value: f"{value:.3f}", na_rep="—"))


if __name__ == "__main__":
    main()
