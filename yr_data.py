import multiprocessing as mp
import datetime
from datetime import timedelta
import re
import requests

"""
Reading from YR's public weather statistics from Trondheim's weather station 688860, Voll
"""

REGEXES = dict(
    time=re.compile(r'<th scope="row">.*<strong>(.*)</strong></th>'),
    temperature=re.compile(r'<td class="temperature .*">(.*)°C</td>'),
    rain=re.compile(r'<td>(.*) mm</td>'),
    humidity=re.compile(r'<td>(.*) %</td>'),
    row=re.compile(r'<tr>'),
    empty=re.compile(r'<td>-</td>')
)


def get_url(location, date):
    return f'https://www.yr.no/place/{location}/almanakk.html?dato={date}'


def get_datetime(date):
    return datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M')


def format_date(date):
    return f'{date.year}-{date.month}-{date.day}'


def add_empty(measurement, column_id):
    if column_id in [3, 4, 5]:
        measurement["temperature"].append(None)

    elif column_id == 6:
        measurement["precipitation"] = None

    elif column_id == 10:
        measurement["humidity"] = None


def retrieve_measurements_by_date(args):
    url, date = args
    measurements, measurement = [], None

    response = requests.get(url, stream=True)
    column = 0
    for line in response.iter_lines():
        if line:
            line = line.decode().strip()

            table_row = REGEXES['row'].search(line)
            column += 1
            column = 0 if table_row else column

            time_match = REGEXES['time'].search(line)
            if time_match:
                measurement = {"temperature": []}
                when = datetime.datetime.strptime(f'{date}T{time_match.group(1)}', '%Y-%m-%dT%H:%M')
                measurement['timestamp'] = int(when.timestamp() * 1e3)
                continue

            temperature_match = REGEXES['temperature'].search(line)
            if temperature_match:
                measurement['temperature'].append(float(temperature_match.group(1)))
                continue

            rain_match = REGEXES['rain'].search(line)
            if rain_match:
                measurement['precipitation'] = float(rain_match.group(1))
                continue

            humidity_match = REGEXES['humidity'].search(line)
            if humidity_match:
                measurement['humidity'] = int(humidity_match.group(1))
                measurement['temperature'] = measurement['temperature'] if len(measurement['temperature']) == 3 else [None]*3
                measurement['temperature'] = dict(
                        measured=measurement['temperature'][0],
                        max=measurement['temperature'][1],
                        min=measurement['temperature'][2]
                    )
                measurements.append(measurement)
                measurement = None
                continue

            empty_row = REGEXES['empty'].search(line)
            add_empty(measurement, column) if empty_row else None

    return {date: measurements}


def retrieve_all_measurements(start_date, end_date, location):
    urls = []
    delta = end_date - start_date

    for day in range(delta.days + 1):
        url_date = start_date + timedelta(days=day)
        urls += [(get_url(location, format_date(url_date)), format_date(url_date))]

    print("No of urls:", len(urls))
    pool = mp.Pool()
    processes = pool.map_async(retrieve_measurements_by_date, urls)
    pool.close()

    return processes.get()


def main():
    location = 'Norway/Tr%C3%B8ndelag/Trondheim/Trondheim/'
    start_date = datetime.datetime(2013, 12, 31)
    end_date = datetime.datetime(2019, 2, 25)

    all_measurements = retrieve_all_measurements(start_date, end_date, location)

    with open("YR_Dataset_Trondheim_2014_2019.json", "w") as f:
        f.write(str(all_measurements))


if __name__ == '__main__':
    main()
