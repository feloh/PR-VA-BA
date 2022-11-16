import pandas as pd
from pyspin.spin import make_spin, Default
from tqdm import tqdm

# Timestamp Limits in Seconds
timeStamp_limits = dict(hr=120, emg=60, PitchZones=30)


# Import Function Excel --> Data
@make_spin(Default, "Reading Excel Sheet...")
def import_data():
    data_frame = pd.read_excel('data/DataSet_270722.xlsx', sheet_name='Data')
    return data_frame


# Take First Function
def take_first(elem):
    return elem[0]


# Linear Gradient Function
def linear_gradient(m, x, b, r):
    y = m * x + b
    if r:
        return round(y)
    else:
        return y


# Smooth Function
def smooth_data(dataframe, column, operation, r):
    filtered = dataframe.index[dataframe[column] == -99].tolist()
    # A Flag contains [Index, Value]
    flags = []
    # Set Flags
    for i in tqdm(filtered):
        if i != filtered[-1]:
            if dataframe.loc[i - 1, column] != -99 and dataframe.loc[i, column] == -99 and dataframe.loc[i + 1, column]\
                    != -99:
                flags.append(tuple([i - 1, dataframe.loc[i - 1, column]]))
                flags.append(tuple([i + 1, dataframe.loc[i + 1, column]]))
            elif dataframe.loc[i - 1, column] != -99 and dataframe.loc[i, column] == -99:
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
    for pair in paired_flags:
        # Steigung
        val_dif = pair[1][1] - pair[0][1]
        # Datenlücke
        time_dif = pair[1][0] - pair[0][0]
        print(time_dif)
        if time_dif <= timeStamp_limits[column]:
            if operation == 'splitGap':
                split_one = time_dif//2
                split_two = time_dif - split_one
                for i in range(split_one):
                    dataframe.loc[pair[0][0] + i, column] = pair[0][1]
                for i in reversed(range(split_two)):
                    dataframe.loc[pair[1][0] - i, column] = pair[1][1]
            elif operation == 'gradient':
                print(time_dif)
                m = val_dif / time_dif
                b = (pair[1][0]*pair[0][1] - pair[0][0]*pair[1][1]) / (pair[1][0] - pair[0][0])
                for i in range(time_dif):
                    dataframe.loc[pair[0][0] + i, column] = linear_gradient(m, i, b, r)

    print(df.loc[df['column'] == -99])


# Smooth: Wenn <= Limit: lineare Steigung von Wert 1 zu Wert 2 bilden und über die Steps verteilen, bei Bedarf runden
# Ansonsten zu große Lücken: Person aus Datensatz aussortieren


# -- Import the Dataset from Excel --
df = import_data()
print('The Data has been successfully imported.')
print(df.info())

# -- Clean the Dataset --
# Remove the BallSpeed Column
df.drop('BallSpeed', axis=1)
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
for ncc in not_combined_columns:
    df.drop(ncc, axis=1)
print('-- Dropped not combined Columns --')

# Group by Participants
participants = df.groupby('ParticipantNumber')
print('-- Grouped Data by Participant-Number --')

# Smooth Objects on Screen
# Smoothing the Pitch Zones values by splitting the missing data gaps into two parts and filling in the first and next
# value
smooth_data(dataframe=df, column='PitchZones', operation='splitGap', r=True)

# TODO Smooth the Pitch Zone Data
## Pitchzonesplate glätten: Grenzwert: 20s
# TODO Smooth the Heart Rate Data
# TODO Smooth the EMG Data
# -- TODO Export and Save the Dataframe --
