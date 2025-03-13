import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

def filter_data(df: pd.DataFrame):
    low_null_columns = ['ICAOCategoryID', 'ICAO_DisplayEng', 'ICAO_DisplayFre', 'SafetyCommIssuedEnum', 'AirportID',
                        'AirportID_AirportName', 'AirportID_CountryID', 'AirportID_CountryID_DisplayEng',
                        'AirportID_CountryID_DisplayFre', 'Airport_ProvinceID', 'AirportID_ProvinceID_DisplayEng',
                        'AirportID_ProvinceID_DisplayFre', 'Location', 'ICAO', 'CommonName', 'OccIncidentTypeID',
                        'OccIncidentTypeID_DisplayEng', 'OccIncidentTypeID_DisplayFre', 'SeriousIncidentEnum',
                        'SeriousIncidentEnum_DisplayEng', 'SeriousIncidentEnum_DisplayFre', 'TsbInvolveID',
                        'TsbInvolveID_DisplayEng', 'TsbInvolveID_DisplayFre', 'TimeZoneID', 'TimeZoneID_DisplayEng',
                        'TimeZoneID_DisplayFre', 'LocationDescription', 'OccID', 'OccNo', 'Latitude', 'LatEnum',
                        'LatEnum_DisplayEng', 'LatEnum_DisplayFre', 'Longitude', 'LongEnum', 'LongEnum_DisplayEng',
                        'LongEnum_DisplayFre', 'CountryID', 'CountryID_DisplayEng', 'CountryID_DisplayFre',
                        'OccClassID',
                        'OccClassID_DisplayEng', 'OccClassID_DisplayFre', 'OccDate', 'OccRegionID',
                        'OccRegionID_DisplayEng',
                        'OccRegionID_DisplayFre', 'OccTime', 'OccTypeID', 'OccTypeID_DisplayEng',
                        'OccTypeID_DisplayFre',
                        'PositionTypeEnum', 'PositionTypeEnum_DisplayEng', 'PositionTypeEnum_DisplayFre', 'ProvinceID',
                        'ProvinceID_DisplayEng', 'ProvinceID_DisplayFre', 'InitTSBNotifDate', 'ReportedByID',
                        'ReportedByID_DisplayEng', 'ReportedByID_DisplayFre', 'RespRegionID', 'RespRegionID_DisplayEng',
                        'RespRegionID_DisplayFre', 'Summary', 'TotalFatalCount', 'TotalMinorCount', 'TotalNoneCount',
                        'TotalSeriousCount', 'TotalUnknownCount', 'NoAircraftInvolved', 'InjuriesEnum',
                        'InjuriesEnum_DisplayEng',
                        'InjuriesEnum_DisplayFre', 'DeployedEnum', 'DeployedEnum_DisplayEng', 'DeployedEnum_DisplayFre']
    return df[low_null_columns]

def output_info(mini_df: pd.DataFrame):
    target_counts = mini_df['TotalFatalCount'].value_counts()
    target_counts.to_csv('outputs/q3_survival_model/target_value_counts.txt', sep='\t', index=True)

def get_model():
    drop_features = ['ICAO_DisplayEng', 'ICAO_DisplayFre', 'AirportID_AirportName', 'AirportID_CountryID_DisplayEng',
                     'AirportID_CountryID_DisplayFre', 'AirportID_ProvinceID_DisplayEng',
                     'AirportID_ProvinceID_DisplayFre',
                     'CommonName', 'OccIncidentTypeID_DisplayEng', 'OccIncidentTypeID_DisplayFre',
                     'SeriousIncidentEnum_DisplayEng',
                     'SeriousIncidentEnum_DisplayFre', 'TsbInvolveID', 'TsbInvolveID_DisplayEng',
                     'TsbInvolveID_DisplayFre',
                     'TimeZoneID_DisplayEng', 'TimeZoneID_DisplayFre', 'LocationDescription', 'OccID', 'OccNo',
                     'Latitude', 'OccDate',
                     'LatEnum_DisplayEng', 'LatEnum_DisplayFre', 'Longitude', 'LongEnum_DisplayEng',
                     'LongEnum_DisplayFre',
                     'CountryID_DisplayEng', 'CountryID_DisplayFre', 'OccClassID_DisplayEng', 'OccClassID_DisplayFre',
                     'OccRegionID_DisplayEng', 'OccRegionID_DisplayFre', 'OccTypeID_DisplayEng', 'OccTypeID_DisplayFre',
                     'PositionTypeEnum_DisplayEng', 'PositionTypeEnum_DisplayFre', 'ProvinceID_DisplayEng',
                     'ProvinceID_DisplayFre', 'ReportedByID_DisplayEng', 'ReportedByID_DisplayFre',
                     'RespRegionID_DisplayEng',
                     'RespRegionID_DisplayFre', 'Summary', 'InjuriesEnum_DisplayEng', 'InjuriesEnum_DisplayFre',
                     'DeployedEnum_DisplayEng', 'DeployedEnum_DisplayFre', 'OccDate']
    numeric_features = ['TotalMinorCount', 'TotalNoneCount', 'TotalSeriousCount', 'TotalUnknownCount']
    binary_features = ['SeriousIncidentEnum', 'NoAircraftInvolved']
    categorical_features = ['SafetyCommIssuedEnum', 'Location', 'OccTime', 'PositionTypeEnum',
                            'InitTSBNotifDate', 'InjuriesEnum', 'DeployedEnum', 'ICAOCategoryID', 'AirportID',
                            'AirportID_CountryID', 'Airport_ProvinceID', 'ICAO', 'OccIncidentTypeID', 'TimeZoneID',
                            'LatEnum', 'LongEnum', 'CountryID', 'OccClassID', 'OccRegionID', 'OccTypeID', 'ProvinceID',
                            'ReportedByID', 'RespRegionID']
    target = 'TotalFatalCount'

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (OneHotEncoder(drop='if_binary'), binary_features),
        (OneHotEncoder(handle_unknown='ignore'), categorical_features)
    )

    pipe_lr = make_pipeline(preprocessor, LogisticRegression(max_iter=1000, random_state=42), memory='outputs/q3_survival_model')

    return pipe_lr