import pandas as pd

from typing import Tuple
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import f1_score, make_scorer, accuracy_score, classification_report

from completeness import report_missing_values, report_summary_statistics, report_record_dates
from categories import get_icao_categories, get_incident_types, explore_data_capture, trend_analysis, zoom_analysis
from model import filter_data, output_info, get_model


""" Returns data dictionary and parsing parameters for ASIS datasets. """
def get_read_params() -> Tuple[dict, list, dict]:
    # load and clean source data dictionary
    data_dict_df = (pd.read_csv('data/data_dict.csv',
                                usecols=['Column name', 'Data type'],
                                encoding='latin1')
                    .rename(columns={'Column name': 'name', 'Data type': 'type'})
                    .replace('varchar', 'object')
                    .replace('char', 'object')
                    .replace('numeric', 'float64')
                    .replace('decimal', 'float64')
                    .replace('smallint', 'Int32')
                    .replace('int', 'Int64')
                    .apply(lambda x: x.strip() if isinstance(x, str) else x)
                    .replace('', 'object')
                    .fillna('object'))

    # add missing occurrence dataset columns
    new_rows = pd.DataFrame({'name': ['AarfCategoryID_DisplayEng', 'AarfCategoryID_DisplayFre'],
                             'type': ['object', 'object']})
    data_dict_df = pd.concat([data_dict_df, new_rows], ignore_index=True)

    # remove occurrence table date/time columns for separate parsing
    datetime_cols = data_dict_df[data_dict_df['type'].isin(['datetime2', 'date', 'time'])]['name'].tolist()
    dtype_dict = dict(zip(data_dict_df.name, data_dict_df.type))
    for col in datetime_cols:
        del dtype_dict[col]
    date_formats = {
        'InitTSBNotifDate': 'HH:mm:ss[.nnnnnnn]',
        'OccDate': 'YYYY-MM-DD',
        'OccTime': 'hh:mm'
    }

    return dtype_dict, datetime_cols, date_formats


""" Returns dataframe loaded with Occurrences csv. """
def load_occurrences() -> pd.DataFrame:
    dtype_dict, datetime_cols, date_formats = get_read_params()
    occurrence_df = (pd.read_csv('data/occurrence.csv',
                                 dtype=dtype_dict,
                                 parse_dates=datetime_cols,
                                 date_format=date_formats)
                     .apply(lambda x: x.strip() if isinstance(x, str) else x)
                     .replace('', pd.NA)
                     .replace('NULL', pd.NA)
                     .replace('null', pd.NA))
    return occurrence_df

""" Creates completeness data quality reports for Occurrence table. """
def report_completeness():
    occurrence_df = load_occurrences()
    report_summary_statistics(occurrence_df)
    report_missing_values(occurrence_df)
    report_record_dates(occurrence_df)

""" Creates ICAO category info for Canadian airports. """
def report_icao_categories():
    occurrence_df = load_occurrences()

    # little cleaning
    occurrence_df['OccDate'] = pd.to_datetime(occurrence_df['OccDate'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    occurrence_df['Month'] = occurrence_df['OccDate'].dt.month
    occurrence_df.fillna({'OccIncidentTypeID': 19, 'OccIncidentTypeID_DisplayEng': 'UNKNOWN',
                          'ICAOCategoryID': 36, 'ICAO_DisplayEng': "UNKNOWN OR UNDETERMINED (UNK)"}, inplace=True)

    # pull some relevant columns
    columns = ['OccID', 'OccNo', 'OccDate', 'OccTime','AirportID_CountryID_DisplayEng', 'AirportID_CountryID',
               'Airport_ProvinceID', 'AirportID_ProvinceID_DisplayEng','AirportID_AirportName', 'AirportID', 'Location',
               'ICAOCategoryID', 'ICAO_DisplayEng', 'OccIncidentTypeID', 'OccIncidentTypeID_DisplayEng']
    subset_df = occurrence_df[columns]
    can_air_df = subset_df[subset_df['AirportID_CountryID'] == 2]

    # do analytics
    get_icao_categories(can_air_df)
    get_incident_types(can_air_df)
    explore_data_capture(can_air_df)
    trend_analysis(can_air_df)
    zoom_analysis(can_air_df)

def survival_model():
    occurrence_df = load_occurrences()

    occurrence_df.fillna({'OccIncidentTypeID': 19, 'OccIncidentTypeID_DisplayEng': 'UNKNOWN',
                          'ICAOCategoryID': 36, 'ICAO_DisplayEng': "UNKNOWN OR UNDETERMINED (UNK)"},
                             inplace=True)

    mini_df = filter_data(occurrence_df)
    output_info(mini_df)

    train_df, test_df = train_test_split(mini_df, test_size=0.2, random_state=42)

    X_train, y_train = (train_df.drop(columns=['TotalFatalCount']), train_df['TotalFatalCount'])
    X_test, y_test = (test_df.drop(columns=['TotalFatalCount']),test_df['TotalFatalCount'])

    custom_scorer = make_scorer(f1_score, average="macro")  # class imbalance

    model = get_model()
    scores = cross_validate(model, X_train, y_train, return_train_score=True, cv=5)
    score_output = f"Cross Validation Results: {scores}"
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    accuracy_output = f"Accuracy score: {accuracy}"
    macro_f1 = f1_score(y_test, predictions, average='macro')
    macro_f1_output = f"Macro F1 score: {macro_f1}"
    class_report = classification_report(y_test, predictions)
    class_report_output = f"Classification report: \n {class_report}"

    with open('outputs/q3_survival_model/model_results.txt', 'w') as file:
        file.write(score_output + '\n')
        file.write(accuracy_output + '\n')
        file.write(macro_f1_output + '\n')
        file.write(class_report_output + '\n')



if __name__ == "__main__":
    report_completeness()
    report_icao_categories()
    survival_model()

