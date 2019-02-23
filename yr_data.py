from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Two ways to get data from Yr.no:

###################################### First way ###############
def getWeather():
    url = 'http://www.yr.no/sted/Norge/Troms/Troms%C3%B8/Troms%C3%B8/varsel_time_for_time.xml'
    r = requests.get(url)
    doc = xmltodict.parse(r.text)

    try:
        weather_data = doc['weatherdata']['observations']['weatherstation'][0]
    except:
        raise ValueError('Could not read weather')

    # weather_data['@lat']

    # weather_data['@lon']

    # weather_data['temperature']['@value']

    # weather_data['temperature']['@time']

    return weather_data


"""

# Its an xml file, each loc has its own url: http://om.yr.no/info/verdata/xml/

This function gets the observations from the closest weather station. The xml also contains the predictions
===========================================================

##################################### Second way ###########################
To get historical data you either have to grab it out of the HTML or use their data science tools

Here is their historical data dumps:
http://thredds.met.no/thredds/metno.html

this is the tool they use
https://drive.google.com/file/d/0B-SaEtrDE91WcEZYcm01VFNjQkk/view

Here is a script to parse the data out of the HTML and into a list of dictionaries. Resolution of the data on these html pages seem to be hourly.
Should be compatible with both python 2 and 3, do pip install six just to be sure, it's a tiny compatibility layer between the two.

"""

from six.moves import urllib

import datetime
import re

URL_FORMAT = 'https://www.yr.no/place/{location}/almanakk.html?dato={date}'

REGEXES = dict(
    time=re.compile(r'<th scope=“row”>.*<strong>(.*)</strong></th>'),
    temperature=re.compile(r'<td class=“temperature .*“>(.*)°C</td>'),
    rain=re.compile(r'<td>(.*) mm</td>'),
    humidity=re.compile(r'<td>(.*) %</td>')
)


def get_url(location, date):
    return URL_FORMAT.format(location=location, date=date)


def get_datetime(date):
    return datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M')


def get_measurements(date, lines):
    messages = []
    message = None

    active = False
    for line in lines:
        line = line.decode()
        line = line.strip()

        time_match = REGEXES['time'].search(line)
        if time_match is not None:
            message = dict(temperature=[])

            when = get_datetime('{date}T{time}'.format(date=date, time=time_match.group(1)))
            message['timestamp'] = int(when.timestamp() * 1e3)
            continue

        temperature_match = REGEXES['temperature'].search(line)
        if temperature_match is not None:
            message['temperature'].append(float(temperature_match.group(1)))
            continue

        rain_match = REGEXES['rain'].search(line)
        if rain_match is not None:
            message['precipitation'] = float(rain_match.group(1))
            continue

        humidity_match = REGEXES['humidity'].search(line)
        if humidity_match is not None:
            message['humidity'] = int(humidity_match.group(1))

            message['temperature'] = dict(
                measured=message['temperature'][0],
                max=message['temperature'][1],
                min=message['temperature'][2]
            )
            messages.append(message)
            message = None
            continue

    return messages


def main():
    location = 'Norway/Akershus/B%C3%A6rum/Fornebu'
    date = '2018-06-03'

    with urllib.request.urlopen(get_url(location=location, date=date)) as url:
        lines = url.readlines()

    print(get_measurements(date, lines))
    return 0


if __name__ == '__main__':
    exit(main())
