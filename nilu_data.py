import sys
import os
import json
import multiprocessing as mp
import requests
from datetime import datetime


def format_url(api, station, component, year, month):
    return f'{api}/obs/historical/{year}-{month}-01/{year}{month+1}-01/{station}?component={component}'


def download_to_json(urls: str) -> dict:
    results = []
    for url in urls:
        req = requests.get(url)
        results += [req.json() if req.status_code == 200 else []]
    return results


def download_nilu_historical_data(args):
    api, station, components, from_date, to_date, savepath = args
    # Compiling list of all urls to gather data from:
    urls = []
    for year in range(from_date.year, to_date.year, 1):
        for month in range(1, 12, 1):
            urls += [format_url(api, station, components, year, month)]

    # Downloading (may take some time):
    results = download_to_json(urls)

    # Filtering empty results:
    results = [result for result in results if len(result) > 0]

    # Joining results together into single dict:
    try:
        result = {
            key: value
            for key, value in results[0][0].items()
            if key not in ["component", "values"]
        }
        air_data = {component: [] for component in components.split(",")}
        for r in results:
            for data in r:
                air_data[data["component"]] += data["values"]
        result["values"] = air_data
    except IndexError:
        result = {}

    # Writing to file:
    filepath = os.path.join(savepath, "nilu", "{}.json".format(station))
    with open(filepath, "w") as f:
        json.dump(result, f)
    print("  - Saved {} data to {}".format(station, filepath))
    return result


def download_to(
    savepath="./data", from_date=datetime(2014, 1, 1), to_date=datetime(2020, 1, 1)
):

    # Check if the data already exists:
    if os.path.isdir(savepath):
        print("Found directory: {}".format(os.path.abspath(savepath)))
    else:
        os.makedirs(savepath, exist_ok=True)
        print("Created directory: {}".format(os.path.abspath(savepath)))

    os.makedirs(os.path.join(savepath, "nilu"), exist_ok=True)

    # Downloading Airquality data from:
    api = "https://api.nilu.no"

    stations = [
        res["station"]
        for res in requests.get(api + "/lookup/stations?area=Trondheim").json()
    ]
    components = ",".join(
        [res["component"] for res in requests.get(api + "/lookup/components").json()]
    )

    print("Downloading air-quality data for {} stations".format(len(stations)))
    packed_args = [
        (api, station_id, components, from_date, to_date, savepath)
        for station_id in stations
    ]
    pool = mp.Pool()
    processes = pool.map_async(download_nilu_historical_data, packed_args)
    pool.close()
    return processes.get()
