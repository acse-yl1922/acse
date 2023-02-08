"""Interactions with rainfall and river data."""

import numpy as np
import pandas as pd
import json
from urllib.request import urlopen

__all__ = ["get_station_data_from_csv","get_live_station_data","get_stationReference_and_position","get_all_live_data"]


def get_station_data_from_csv(filename, station_reference):
    """Return readings for a specified recording station from .csv file.

    Parameters
    ----------

    filename: str
        filename to read
    station_reference
        station_reference to return.

    >>> data = get_station_data_from_csv('resources/wet_day.csv')
    """
    frame = pd.read_csv(filename)
    frame = frame.loc[frame.stationReference == station_reference]

    return pd.to_numeric(frame.value.values)

def get_stationReference_and_position(url):
    """Return list of station reference and its position data

    Parameters
    ----------
    >>> data = get_stationReference_and_position_data
    """
    web=urlopen(url)

    data_json=json.loads(web.read())
    station_data=pd.json_normalize(data_json["items"])
    return station_data

def get_live_station_data(url, station_reference):
    """Return readings for a specified recording station from live API.

    Parameters
    ----------

    url: str
        a web link to read
    station_reference
        station_reference to return.

    >>> data = get_live_station_data('0184TH')
    """
    station_data=get_stationReference_and_position_data(url)
    s_data=station_data[station_data["stationReference"]==station_reference]
    lat=s_data["lat"].iloc[0]
    lon=s_data["long"].iloc[0]
    latitude=[]
    longitude=[]
    stationReference=[]
    parameter=[]
    dateTime=[]
    qualifier=[]
    value=[]
    unitName=[]
    for i in s_data["measures"].iloc[0]:
        tmp_data=json.loads(urlopen(i["@id"]).read())["items"]
        parameter.append(tmp_data["parameter"])
        qualifier.append(tmp_data["qualifier"])
        unitName.append(tmp_data["unitName"])
        dateTime.append(tmp_data["latestReading"]["dateTime"])
        value.append(tmp_data["latestReading"]["value"])
        stationReference.append(station_reference)
        latitude.append(lat)
        longitude.append(lon)
    df=pd.DataFrame({"dateTime":dateTime,"stationReference":stationReference,
                  "latitude":latitude,"longitude":longitude,"parameter":parameter,
                  "qualifier":qualifier,"unitName":unitName,"value":value})
    return df

def get_all_live_data(url="https://environment.data.gov.uk/flood-monitoring/id/stations/"):
    station_data=get_stationReference_and_position(url)
    station_data=station_data[station_data["measures"].apply(lambda x: type(x)!=float)]
    latitude=[]
    longitude=[]
    stationReference=[]
    parameter=[]
    dateTime=[]
    qualifier=[]
    value=[]
    unitName=[]
    for i in range(station_data.shape[0]):
        lat=station_data.iloc[i]["lat"]
        lon=station_data.iloc[i]["long"]
        station_reference=station_data.iloc[i]["stationReference"]
        for j in station_data.iloc[i]["measures"]:
            tmp_data=json.loads(urlopen(j["@id"]).read())["items"]
            if "latestReading" not in tmp_data:
                continue
            elif isinstance(tmp_data["latestReading"],str):
                continue
            else:
                parameter.append(tmp_data["parameter"])
                qualifier.append(tmp_data["qualifier"])
                unitName.append(tmp_data["unitName"])
                dateTime.append(tmp_data["latestReading"]["dateTime"])
                value.append(tmp_data["latestReading"]["value"])
                stationReference.append(station_reference)
                latitude.append(lat)
                longitude.append(lon)

        else:
            continue
    df=pd.DataFrame({"dateTime":dateTime,"stationReference":stationReference,
                  "latitude":latitude,"longitude":longitude,"parameter":parameter,
                  "qualifier":qualifier,"unitName":unitName,"value":value})
    df.to_csv('flood_tool/resources/latest_live_data.csv',index=False)