# clean_data.py
"""
TJ Hart and Adam Ward
Math 402 Final Project
12/12/24
"""

import pandas as pd
import re


def check_weather(column: pd.Series, key: str)->pd.Series:
    """
    Helper function to classify vague weather types into one of seven general categories.
    
    Parameters:
        column (pd.Series): Column of data to classified
        key (str):          One of the seven general categories according to which the column 
                                will be one-hot encoded

    Returns:
        (pd.Series):  A new one-hot encoded column of data points that match the given key

    """
    # define a dictionary mapping the general categories to the Regex code that will apply to that category 
    weather_dict = {'snow/hail': 'snow|hail|wintry|freez', 
                    'rain': 'rain|mist|shower|drizzle',
                    'fog': 'fog',
                    'dust/smoke': 'dust|smoke|haze', 
                    'storm': 'storm|thunder',
                    'cloudy': 'cloud|overcast',
                    'fair': 'fair|clear', 
                    'other': 'N/A|squalls'}
    
    # now, return the one-hot encoded column by comparing the string in each entry
    # with the given Regex code and placing a True if it matches and False otherwise

    return column.str.contains(weather_dict[key], flags=re.IGNORECASE, regex=True)


def clean_data(df: pd.DataFrame, to_drop: list=['Unnamed: 0', 'End_Time', 'County', 'State',
                                                 'Country', 'Timezone', 'Airport_Code',
                                                 'Weather_Timestamp', 'Wind_Direction', 
                                                 'Nautical_Twilight', 'Astronomical_Twilight']):
    """
    Clean an accident dataset. Accepts a dataframe containing the accident data and 
    returns a cleaned version.

    Parameters:
        df (pd.Dataframe):  Accident dataset
        to_drop (list):     Columns to be dropped from df

    Returns:
        df (pd.Dataframe):  Cleaned accident dataset
    """
    # start cleaning by dropping the specified columns
    df.drop(columns=set(to_drop).intersection(set(df.columns)), inplace=True)

    # create time splits for start time of the accident
    if 'Start_Time' in df.columns:
        df["Start_Time"] = pd.to_datetime(df["Start_Time"], format='mixed')
        df["Year"] = df["Start_Time"].dt.year
        df["Month"] = df["Start_Time"].dt.month
        df["Day"] = df["Start_Time"].dt.day_of_week
        df["Hour"] = df["Start_Time"].dt.hour + (df["Start_Time"].dt.minute / 60)
        df.drop(columns=["Start_Time"], inplace=True)
    else:
        print("No Start_Time Column")

    # drop the end time column
    if 'End_Time' in df.columns:
        df.drop(columns=["End_Time"], inplace=True)
    else:
        print("No End_Time Column")

    # run through the columns and clean them accordingly
    for column in df.columns:
        # fill these values with "Unspecified"
        if column == "Sunrise_Sunset" or column == "Civil_Twilight":
            df[column].fillna("Unspecified", inplace=True)

        # fill these values with the Start_Lat and End_Lat, respectively
        elif column == "End_Lat":
            mask = df[column].isna()
            df.loc[mask, column] = df.loc[mask, ("Start_Lat")]

        elif column == "End_Lng":
            mask = df[column].isna()
            df.loc[mask, column] = df.loc[mask, ("Start_Lng")]

        # fill these values with the median value for the given month
        elif column in ['Temperature(F)', 
                        'Wind_Chill(F)', 
                        'Humidity(%)', 
                        'Pressure(in)', 
                        'Visibility(mi)', 
                        'Wind_Speed(mph)', 
                        'Precipitation(in)']:

            # a little from stack overflow on this, but the idea is totally ours
            df[column] = df[column].fillna(df.groupby('Month')[column].transform('median')) 

        # fill these values with the mode value for the given month
        elif column == 'Weather_Condition':
            # again, some help from stack overflow here
            df[column] = df[column].fillna(df.groupby('Month')[column].transform(lambda x: x.mode()[0])) 

            # Now, loop through the different groupings of weather conditions we have chosen
            weather_types = ['snow/hail', 'rain', 'fog', 'dust/smoke', 'storm', 'cloudy', 'fair', 'other']

            # create a new one-hot encoded column for each grouping using our helper function
            for key in weather_types:
                df[key] = check_weather(df['Weather_Condition'], key)
            
            # drop the old column now that we have created the new ones
            df.drop(columns=column, inplace=True)

        else:
            pass

    return df
