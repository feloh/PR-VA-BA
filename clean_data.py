import pandas as pd
from pyspin.spin import make_spin, Default
import numpy as np
from tqdm import tqdm
import gc


# Import Function Excel --> Data
@make_spin(Default, "Reading Excel Sheet...")
def import_data(path, name):
    print('Path: ', path)
    data_frame = pd.read_excel(path, sheet_name=name, dtype={'BallPosession': np.int64, 'MatchTime': np.int64,
                                                             'BallSpeed': np.int64, 'ReplaysTotal': np.int64,
                                                             'GameInterruptionsDiverse': np.int64})
    return data_frame


# Take First Function
def take_first(elem):
    return elem[0]


# Smooth Function
def smooth_data(dataframe, column, operation, r, timestamp_limit):
    filtered = dataframe.index[dataframe[column] == -99].tolist()
    # A Flag contains [Index, Value]
    flags = []
    # Set Flags
    for i in tqdm(filtered):
        if i != (len(dataframe.index) - 1):
            if i != 0 and dataframe.loc[i - 1, column] != -99 and dataframe.loc[i, column] == -99 and dataframe.loc[i + 1, column]\
                    != -99:
                flags.append(tuple([i - 1, dataframe.loc[i - 1,  column]]))
                flags.append(tuple([i + 1, dataframe.loc[i + 1, column]]))
            elif i != 0 and dataframe.loc[i - 1, column] != -99 and dataframe.loc[i, column] == -99:
                flags.append(tuple([i - 1, dataframe.loc[i - 1, column]]))
            elif dataframe.loc[i + 1, column] != -99 and dataframe.loc[i, column] == -99:
                flags.append(tuple([i + 1, dataframe.loc[i + 1, column]]))
        else:
            flags.append(flags[-1])

    flags.sort(key=take_first)
    # Paired Flags contain the first and the last Flag of a missing data part
    paired_flags = list(zip(flags[0::2], flags[1::2]))
    print(len(paired_flags), ' paired Flags')
    del flags
    gc.collect()
    if len(paired_flags) != 0:
        # Smooth out gaps in dependency of the Data and the Data Type
        print('Correcting the values in column ', column)
        for pair in tqdm(paired_flags):
            # Steigung
            val_dif = pair[1][1] - pair[0][1]
            # Datenlücke
            time_dif = pair[1][0] - pair[0][0]
            print(time_dif)

            if time_dif <= timestamp_limit:
                if operation == 'splitGap':
                    split_one = time_dif//2
                    split_two = time_dif - split_one
                    for i in range(split_one):
                        dataframe.at[pair[0][0] + i + 1, column] = pair[0][1]
                    for i in reversed(range(split_two)):
                        dataframe.at[pair[1][0] - i - 1, column] = pair[1][1]
                elif operation == 'gradient':
                    m = val_dif / time_dif
                    for i in range(time_dif):
                        if r:
                            dataframe.at[pair[0][0] + i + 1, column] = round(pair[0][1] + m * (i + 1))
                        else:
                            dataframe.at[pair[0][0] + i + 1, column] = pair[0][1] + m * (i + 1)
    del paired_flags
    gc.collect()


data_path = 'data/DataSet_270722.xlsx'
data_sheet_name = 'Data'
Data_to_Smooth = [
    dict(name='HR', operation='gradient', timeStampLimit=120, r=True),
    dict(name='EMGDelta', operation='gradient', timeStampLimit=60, r=False),
    dict(name='Zone1', operation='splitGap', timeStampLimit=30, r=True),
    dict(name='Zone2', operation='splitGap', timeStampLimit=30, r=True),
    dict(name='Zone3', operation='splitGap', timeStampLimit=30, r=True),
    dict(name='Zone4', operation='splitGap', timeStampLimit=30, r=True),
    dict(name='Zone5', operation='splitGap', timeStampLimit=30, r=True),
    dict(name='Zone6', operation='splitGap', timeStampLimit=30, r=True),
    dict(name='ObjectsOnScreen', operation='splitGap', timeStampLimit=30, r=True),
    dict(name='BallPosessionHome', operation='splitGap', timeStampLimit=30, r=True),
    dict(name='BallPosessionAway', operation='splitGap', timeStampLimit=30, r=True),
    dict(name='OddsDiffSquare', operation='gradient', timeStampLimit=300, r=True)
]
combined_columns = [
    'BallPosession',
    'KickOff',
    'GoalKeepKick',
    'CornerKick',
    'FreeKick',
    'ThrowIn',
    'Foul',
    'Injury',
    'YellowORRedCard',
    'Goal',
    'GoalShoot',
    'PossenChangeHomeAway',
    'PitchZones',
    'GoalDiff1bis3'
]
export_path = 'data/Output.xlsx'
export_path_csv = 'data/Output.csv'

# -- Import the Dataset from Excel --
df = import_data(data_path, data_sheet_name)
print('The Data has been successfully imported.')
print(df.info())

# -- Clean the Dataset --

# Remove the BallSpeed Column
df = df.drop(columns='BallSpeed')
print('-- Dropped BallSpeed --')

# Remove  combined Columns
df = df.drop(columns=combined_columns)
print('-- Dropped not combined Columns --')
del combined_columns
gc.collect()

# Smooth Missing Data
for data in Data_to_Smooth:
    smooth_data(dataframe=df, column=data['name'], operation=data['operation'],
                timestamp_limit=data['timeStampLimit'], r=data['r'])


# Drop Rows with missing Data
print('Searching for missing Data...')
for data in tqdm(Data_to_Smooth):
    df = df.drop(index=df.index[df[data['name']] == -99])

# -- Export and Save the Dataframe --
print('Export the Data to', export_path)
df.to_excel(export_path, index=False)
df.to_csv(export_path_csv, header=False, index=False)
