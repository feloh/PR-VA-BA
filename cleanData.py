import pandas as pd
from pyspin.spin import make_spin, Default
from tqdm import tqdm


# Import Function Excel --> Data
@make_spin(Default, "Reading Excel Sheet...")
def import_data(path, name):
    print('Path: ', path)
    data_frame = pd.read_excel(path, sheet_name=name)
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
                flags.append(tuple([i - 1, dataframe.loc[i - 1, column]]))
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

    # Smooth out gaps in dependency of the Data and the Data Type
    print('Correcting the values in column ', column)
    for pair in tqdm(paired_flags):
        # Steigung
        val_dif = pair[1][1] - pair[0][1]
        # Datenlücke
        time_dif = pair[1][0] - pair[0][0]

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


data_path = 'data/DataSet_270722.xlsx'
data_sheet_name = 'Data'
Data_to_Smooth = [
    dict(name='HR', operation='gradient', timeStampLimit=300, r=True),
    dict(name='EMGDelta', operation='gradient', timeStampLimit=300, r=False),
    dict(name='PitchZones', operation='splitGap', timeStampLimit=300, r=True),
    dict(name='ObjectsOnScreen', operation='splitGap', timeStampLimit=300, r=True)
]
export_path = 'data/Output.xlsx'

# -- Import the Dataset from Excel --
df = import_data(data_path, data_sheet_name)
print('The Data has been successfully imported.')
print(df.info())

# -- Clean the Dataset --
# Remove the BallSpeed Column
df = df.drop(columns='BallSpeed')
print('-- Dropped BallSpeed --')

# Remove not combined Columns
not_combined_columns = [
    'BallPosessionAway',
    'BallPosessionHome',
    'CornerKickAway',
    'CornerKickHome',
    'FoulAway',
    'FoulHome',
    'FreeKickAway',
    'FreeKickHome',
    'GoalDiff1',
    'GoalDiff2',
    'GoalDiff3',
    'GoalKeeperKickAway',
    'GoalKeeperKickHome',
    'GoalShootAway',
    'GoalShootHome',
    'KickOffAway',
    'KickOffHome',
    'PossenChangeHometoAway',
    'PossesionChangeAwaytoHome',
    'ThrowInAway',
    'ThrowInHome',
    'YellowOrRedCardAway',
    'YellowOrRedCardHome',
    'Zone1',
    'Zone2',
    'Zone3',
    'Zone4',
    'Zone5',
    'Zone6',
]
df = df.drop(columns=not_combined_columns)
print('-- Dropped not combined Columns --')

# Smooth Missing Data
for data in Data_to_Smooth:
    smooth_data(dataframe=df, column=data['name'], operation=data['operation'], r=data['r'],
                timestamp_limit=data['timeStampLimit'])

# Drop Participants with missing Data
participant_id = []
print('Searching for missing Data...')
for data in tqdm(Data_to_Smooth):
    for row in df.index[df[data['name']] == -99]:
        print(data['name'])
        participant_id.append(df.at[row, 'ParticipantNumber'])
set_pi = set(participant_id)
unique_participant_id = list(set_pi)
for upi in unique_participant_id:
    print('Dropping Participant ', upi, 'for lack of Data.')
    df = df.drop(df.index[df['ParticipantNumber'] == upi])

# -- Export and Save the Dataframe --
print('Export the Data to', export_path)
df.to_excel(export_path)
