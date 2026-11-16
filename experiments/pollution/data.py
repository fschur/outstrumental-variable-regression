"""Build the five-metro panel from NMMAPS, Census, EPA, ERA5-Land and airport observations."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import urlretrieve

import pandas as pd


NMMAPS = "https://raw.githubusercontent.com/jhu-advdatasci/2018/6effedaf245d531f9096863260d867cf583192a6/data/NMMAPS_storm/mortality/"
CENSUS = "https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-tot.csv"
YEARS = range(2000, 2006)
# NMMAPS code, city-centre coordinates, local time zone, primary and backup airports.
METROS = [
    ("la",   34.05, -118.24, "America/Los_Angeles", "LAX", "BUR"),
    ("ny",   40.71,  -74.01, "America/New_York",    "LGA", None),
    ("chic", 41.88,  -87.63, "America/Chicago",     "MDW", "ORD"),
    ("dlft", 32.78,  -96.80, "America/Chicago",     "DAL", "DFW"),
    ("hous", 29.76,  -95.37, "America/Chicago",     "HOU", None),
]


def download(cache, filename, url):
    path = cache / filename
    if not path.exists():
        with TemporaryDirectory(dir=cache) as tmp:
            partial = Path(tmp) / filename
            urlretrieve(url, partial)
            partial.replace(path)
    return path


def mortality(cache, counties):
    deaths = pd.read_csv(download(cache, "mortality.csv.bz2", NMMAPS + "mortality_1987-2005.csv.bz2"),
                         usecols=["city", "date", "death"], parse_dates=["date"])
    deaths = deaths.loc[deaths.city.isin(counties.city) & deaths.date.dt.year.isin(YEARS)].copy()
    deaths["year"] = deaths.date.dt.year
    pop = pd.read_csv(download(cache, "population.csv", CENSUS), encoding="latin1",
                      dtype={"STATE": str, "COUNTY": str})
    pop["fips"] = pop.STATE.str.zfill(2) + pop.COUNTY.str.zfill(3)
    pop = counties.merge(pop, on="fips", validate="one_to_one")
    if len(pop) != len(counties):
        raise ValueError("Missing Census population for an NMMAPS county.")
    annual = pop.groupby("city")[[f"POPESTIMATE{year}" for year in YEARS]].sum()
    annual.columns = pd.Index(YEARS, name="year")
    return deaths.merge(annual.stack().rename("population"), on=["city", "year"], validate="many_to_one")


def pollution(cache, counties):
    columns = ["State Code", "County Code", "Site Num", "POC", "Date Local", "Observation Percent",
               "Arithmetic Mean", "Units of Measure", "Event Type", "Sample Duration"]
    daily = []
    for year in [1999, *YEARS]:  # December 31 supplies the lag for January 1.
        name = f"daily_88101_{year}.zip"
        values = pd.read_csv(download(cache, name, "https://aqs.epa.gov/aqsweb/airdata/" + name),
                             usecols=columns, dtype={"State Code": str, "County Code": str},
                             keep_default_na=False, parse_dates=["Date Local"])
        values["fips"] = values["State Code"] + values["County Code"]
        values = values.merge(counties[["fips", "city"]], on="fips", validate="many_to_one")
        values = values.loc[values["Event Type"].isin(["None", "Included"])
                            & (values["Observation Percent"] >= 75) & (values["Arithmetic Mean"] >= 0)
                            & values["Sample Duration"].isin(["24 HOUR", "24-HR BLK AVG"])]
        if not values["Units of Measure"].eq("Micrograms/cubic meter (LC)").all():
            raise ValueError("Unexpected units for EPA PM2.5.")
        monitors = values.groupby(["city", "Date Local", "fips", "Site Num", "POC"])["Arithmetic Mean"].mean()
        sites = monitors.groupby(level=[0, 1, 2, 3]).mean()  # first average collocated monitors
        daily.append(sites.groupby(level=[0, 1]).mean().unstack("city"))
    pm = pd.concat(daily).sort_index().asfreq("D")
    return ((pm + pm.shift()) / 2).rename_axis(index="date").stack().rename("pm25").reset_index()


def gridded_weather(cache):
    query = urlencode({
        "latitude": ",".join(str(m[1]) for m in METROS), "longitude": ",".join(str(m[2]) for m in METROS),
        "timezone": ",".join(m[3] for m in METROS), "start_date": "2000-01-01", "end_date": "2005-12-31",
        "models": "era5_land", "daily": "temperature_2m_mean,relative_humidity_2m_mean",
    })
    path = download(cache, "era5-land.json", "https://archive-api.open-meteo.com/v1/archive?" + query)
    frames = []
    for (city, lat, lon, zone, _, _), site in zip(METROS, json.loads(path.read_text())):
        if site["timezone"] != zone or abs(site["latitude"] - lat) >= .11 or abs(site["longitude"] - lon) >= .11:
            raise ValueError(f"Unexpected ERA5-Land grid point for {city}.")
        frames.append(pd.DataFrame(site["daily"]).assign(city=city))
    weather = pd.concat(frames).rename(columns={"time": "date", "temperature_2m_mean": "temperature",
                                                "relative_humidity_2m_mean": "humidity"})
    weather["date"] = pd.to_datetime(weather.date)
    return weather


def airport_weather(cache, station, zone):
    query = urlencode({"station": station, "data": ["tmpf", "dwpf", "wxcodes"], "year1": 2000, "month1": 1,
                       "day1": 1, "year2": 2006, "month2": 1, "day2": 2, "tz": "Etc/UTC", "format": "onlycomma",
                       "latlon": "no", "elev": "no", "missing": "M", "trace": "T", "direct": "no",
                       "report_type": 3}, doseq=True)
    path = download(cache, f"{station}.csv", "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + query)
    hourly = pd.read_csv(path, na_values="M")
    hour = pd.to_datetime(hourly.valid, utc=True).dt.floor("h").dt.tz_convert(zone)
    hourly["rain"] = hourly.wxcodes.str.contains("RA|DZ", na=False)
    hourly["snow"] = hourly.wxcodes.str.contains("SN|SG", na=False)
    hourly = hourly.groupby(hour).agg({"tmpf": "mean", "dwpf": "mean", "rain": "max", "snow": "max"}).dropna()
    days = hourly.groupby(hourly.index.tz_localize(None).normalize())
    daily = days.agg({"tmpf": "mean", "dwpf": "mean", "rain": "max", "snow": "max"}).loc[days.size() >= 18]
    # Round before LOESS so identical temperatures receive identical neighborhoods.
    daily[["tmpf", "dwpf"]] = ((daily[["tmpf", "dwpf"]] - 32) * 5 / 9).round(3)
    return daily.rename(columns={"tmpf": "air_temp", "dwpf": "dewpoint"})


def metro_days(cache):
    cache.mkdir(parents=True, exist_ok=True)
    crosswalk = pd.read_csv(download(cache, "counties.csv", NMMAPS + "nmmaps_cities.csv"), dtype={"fips": str})
    counties = crosswalk.loc[crosswalk.city.isin([m[0] for m in METROS])]
    data = mortality(cache, counties).merge(gridded_weather(cache), on=["city", "date"], validate="one_to_one")
    data = data.merge(pollution(cache, counties), how="left", on=["city", "date"], validate="one_to_one")
    data = data.sort_values(["city", "date"])
    data["temp4"] = data.groupby("city").temperature.transform(lambda s: s.rolling(4).mean())
    # Intersect city-specific central 90% ranges. Keep each two-day exposure within one calendar month.
    limits = data.groupby("city").pm25.quantile([.05, .95]).unstack()
    data = data.loc[data.pm25.between(limits[.05].max(), limits[.95].min()) & data.temp4.notna()
                    & (data.date.dt.day > 1)]
    frames = []
    for city, _, _, zone, primary, backup in METROS:
        rows = data.loc[data.city == city]
        weather = airport_weather(cache, primary, zone).reindex(rows.date)
        if backup and weather.isna().any().any():
            weather = weather.combine_first(airport_weather(cache, backup, zone).reindex(rows.date))
        if weather.isna().any().any():
            raise ValueError(f"Incomplete airport weather in {city}.")
        frames.append(rows.join(weather, on="date", validate="one_to_one"))
    panel = pd.concat(frames)
    panel["city"] = panel.city.map(counties.set_index("city").cityname.to_dict())
    return panel.sort_values(["city", "date"]).reset_index(drop=True)
