import pandas as pd
from pyspin.spin import make_spin, Default

# Timestamp Limits in Seconds
timeStamp_limits = dict(hr=120, emg=60, pz=20)


# Import Function Excel --> Data
@make_spin(Default, "Reading Excel Sheet...")
def import_data():
    data_frame = pd.read_excel('data/DataSet_270722.xlsx', sheet_name='Data')
    return data_frame


# Smooth Function
def smooth_data(dataframe, column, mw):
    rslt_df = dataframe.loc[dataframe[column] == -99]
    rslt_df = rslt_df.sort_index()
    print(rslt_df)
    if mw:
        print('mw')
    else:
        print('smooth')


# Spalte Stepper: Wenn Fehlwert, dann nimm den Wert davor und geh solange die Spalte entlang, bis kein Fehlwert und
# nimm den nächsten wert dazu Davon Werte und Index in Array Steigung aus erstem und letztem Wert. Werte
# dazwischen nach Index und Steigung generieren --> Runden wenn mw true, dann nur mittelwert bilden # wenn
# Grenzwert überschritten dann ???


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
grouped = df.groupby('ParticipantNumber')
print('-- Grouped Data by Participant-Number --')

# Smooth Objects on Screen
smooth_data(dataframe=df, column='PitchZones', mw=False)

# TODO Smooth the Pitch Zone Data
## Pitchzonesplate glätten: Grenzwert: 20s
# TODO Smooth the Heart Rate Data
# TODO Smooth the EMG Data
# -- TODO Export and Save the Dataframe --
