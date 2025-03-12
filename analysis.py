from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

""" Creates reports of column null value ratios. """
def report_missing_values(df: pd.DataFrame):
    # histogram of null value rates
    null_value_rates = round(df.isnull().mean() * 100, 2)
    null_value_rates.plot.hist(bins=10)
    plt.title("Histogram of Column Null Value Rates")
    plt.xlabel("Null Value Rate (NULL/Total*100)")
    plt.ylabel("Column Frequency (Count)")
    plt.savefig('outputs/q1_completeness/null_value_distribution.png')
    plt.clf()

    # plot of more complete columns
    missing_percentages = round((df.isnull().sum() / df.shape[0]) * 100, 2)
    very_high_missing = missing_percentages[missing_percentages < 5]
    very_high_missing.plot(kind='bar')
    plt.title('Columns with Low (< 5%) Null Value Rates')
    plt.ylabel('Null Value Rate (NULL/Total*100)')
    plt.xlabel('Column Name')
    plt.savefig('outputs/q1_completeness/low_null_value_rates.png')
    plt.clf()

    # output lists of null value rates and info
    columns_with_100_nulls = null_value_rates[null_value_rates == 100].index.tolist()
    c_100 = f"Columns with 100% null value rates: {columns_with_100_nulls}."
    count_100 = f"100% count: {len(columns_with_100_nulls)}"
    b_90 = null_value_rates.between(90, 100, inclusive='left')
    columns_90 = f"Columns with 90-100% null value rates: {b_90[b_90].index.tolist()}"
    count_90 = f"90% count: {len(b_90[b_90].index.tolist())}"
    b_80 = null_value_rates.between(80, 90, inclusive='left')
    columns_80 = f"Columns with 80-90% null value rates: {b_80[b_80].index.tolist()}"
    count_80 = f"80% count: {len(b_80[b_80].index.tolist())}"
    b_70 = null_value_rates.between(70, 80, inclusive='left')
    columns_70 = f"Columns with 70-80% null value rates: {b_70[b_70].index.tolist()}"
    count_70 = f"70% count: {len(b_70[b_70].index.tolist())}"
    b_60 = null_value_rates.between(60, 70, inclusive='left')
    columns_60 = f"Columns with 60-70% null value rates: {b_60[b_60].index.tolist()}"
    count_60 = f"60% count: {len(b_60[b_60].index.tolist())}"
    b_50 = null_value_rates.between(50, 60, inclusive='left')
    columns_50 = f"Columns with 50-60% null value rates: {b_50[b_50].index.tolist()}"
    count_50 = f"50% count: {len(b_50[b_50].index.tolist())}"
    b_40 = null_value_rates.between(40, 50, inclusive='left')
    columns_40 = f"Columns with 40-50% null value rates: {b_40[b_40].index.tolist()}"
    count_40 = f"40% count: {len(b_40[b_40].index.tolist())}"
    b_30 = null_value_rates.between(30, 40, inclusive='left')
    columns_30 = f"Columns with 30-40% null value rates: {b_30[b_30].index.tolist()}"
    count_30 = f"30% count: {len(b_30[b_30].index.tolist())}"
    b_20 = null_value_rates.between(20, 30, inclusive='left')
    columns_20 = f"Columns with 20-30% null value rates: {b_20[b_20].index.tolist()}"
    count_20 = f"20% count: {len(b_20[b_20].index.tolist())}"
    b_10 = null_value_rates.between(10, 20, inclusive='left')
    columns_10 = f"Columns with 10-20% null value rates: {b_10[b_10].index.tolist()}"
    count_10 = f"10% count: {len(b_10[b_10].index.tolist())}"
    b_0 = null_value_rates.between(0, 10, inclusive='left')
    columns_0 = f"Columns with 0-10% null value rates: {b_0[b_0].index.tolist()}"
    count_0 = f"0% count: {len(b_0[b_0].index.tolist())}"
    total_null_count = f"Total null count: {df.isnull().sum().sum()}"
    total_cells = f"Total cells: {df.size}"
    empty_cell_percent = f"Empty cell percentage: {(df.isnull().sum().sum()/df.size) * 100}"
    total_column_count = f"Total column count: {df.shape[1]}"
    total_row_count = f"Total row count: {df.shape[0]}"

    with open('outputs/q1_completeness/nulls_output.txt', 'w') as file:
        file.write(c_100 + '\n')
        file.write(count_100 + '\n\n')
        file.write(columns_90 + '\n')
        file.write(count_90 + '\n\n')
        file.write(columns_80 + '\n')
        file.write(count_80 + '\n\n')
        file.write(columns_70 + '\n')
        file.write(count_70 + '\n\n')
        file.write(columns_60 + '\n')
        file.write(count_60 + '\n\n')
        file.write(columns_50 + '\n')
        file.write(count_50 + '\n\n')
        file.write(columns_40 + '\n')
        file.write(count_40 + '\n\n')
        file.write(columns_30 + '\n')
        file.write(count_30 + '\n\n')
        file.write(columns_20 + '\n')
        file.write(count_20 + '\n\n')
        file.write(columns_10 + '\n')
        file.write(count_10 + '\n\n')
        file.write(columns_0 + '\n')
        file.write(count_0 + '\n\n')
        file.write(total_null_count + '\n')
        file.write(total_cells + '\n')
        file.write(empty_cell_percent + '\n')
        file.write(total_column_count + '\n')
        file.write(total_row_count)

""" Creates csv with general summary statistics. """
def report_summary_statistics(df: pd.DataFrame):
    # summarize interesting features
    interesting_df = df[['AirportID_CountryID', 'ICAO', 'CountryID', 'Distance', 'ICAOCategoryID',
                         'OccClassID', 'OccDate', 'OccIncidentTypeID', 'OccRegionID', 'OccTime', 'OccTypeID',
                         'SeriousIncidentEnum', 'Summary', 'TotalFatalCount', 'TotalMinorCount', 'TotalNoneCount',
                         'TotalSeriousCount', 'TotalUnknownCount']]
    interesting_df.describe().to_csv('outputs/q1_completeness/data_summary.csv', index=True)

""" Creates record frequency reports for Occurrences table. """
def report_record_dates(df: pd.DataFrame):
    # frequency per year over history of dataset
    df['OccDate'] = pd.to_datetime(df['OccDate'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df['Year'] = df['OccDate'].dt.year
    event_frequency_by_year = df['Year'].value_counts().sort_index()
    event_frequency_by_year.plot(kind='bar', figsize=(10, 6))
    plt.title('Occurrence Frequency - Entire Dataset')
    plt.xlabel('Event Year')
    plt.ylabel('Event Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/q1_completeness/freq_per_year.png')
    plt.clf()

    # frequency per month over last 3 years
    df['Month'] = df['OccDate'].dt.month
    df_filtered_recent = df[df['Year'].isin([2023, 2024, 2025])]
    event_counts = df_filtered_recent.pivot_table(index='Month', columns='Year', aggfunc='size', fill_value=0)
    event_counts.plot(kind='bar', figsize=(10, 6))
    plt.title('Occurrence Frequency - Last 3 Years')
    plt.xlabel('Event Month')
    plt.ylabel('Number of Events')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.savefig('outputs/q1_completeness/rec_freq_per_month.png')
    plt.clf()

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

""" Generates output to capture incident type information. """
def get_incident_types(df: pd.DataFrame):
    # get unique values for incident types
    unique_inc_type = f"Unique incident types: {df['OccIncidentTypeID'].unique()}"
    unique_inc_type_count = f"Unique types count: {len(df['OccIncidentTypeID'].unique())}"
    df_inc_desc = df.drop_duplicates(subset=['OccIncidentTypeID', 'OccIncidentTypeID_DisplayEng'])

    with open('outputs/q2_icao_categories/incident_info.txt', 'w') as file:
        file.write(unique_inc_type + '\n')
        file.write(unique_inc_type_count + '\n')

    df_inc_desc.to_csv('outputs/q2_icao_categories/inc_map.txt', sep='\t', index=False)

""" Generates output to capture ICAO category information. """
def get_icao_categories(df: pd.DataFrame):
    # get unique values for ICAO categories
    unique_icao_cat = f"Unique ICAO categories: {df['ICAOCategoryID'].unique()}"
    unique_icao_cat_count = f"Unique categories count: {len(df['ICAOCategoryID'].unique())}"
    df_cat_desc = df.drop_duplicates(subset=['ICAOCategoryID', 'ICAO_DisplayEng'])

    with open('outputs/q2_icao_categories/category_info.txt', 'w') as file:
        file.write(unique_icao_cat + '\n')
        file.write(unique_icao_cat_count + '\n')

    df_cat_desc.to_csv('outputs/q2_icao_categories/cat_map.txt', sep='\t', index=False)

""" Generates output for some data capture information. """
def explore_data_capture(df: pd.DataFrame):
    row_count = f"Total row count: {df.shape[0]}"
    combo = f"Count unique combos OccID and ICAOCategoryID: {df.drop_duplicates(subset=['OccID', 'ICAOCategoryID']).shape[0]}"
    dates = f"Count of unique combos OccDate and OccNo: {df.drop_duplicates(subset=['OccDate', 'OccNo']).shape[0]}"
    dates2 = f"Count of unique combos of OccDate and OccID: {df.drop_duplicates(subset=['OccDate', 'OccID']).shape[0]}"
    occid = f"Count of unique OccID values: {df['OccID'].nunique()}"
    occno = f"Count of unique OccNo values: {df['OccNo'].nunique()}"
    null_air = f"Count of null values in 'AirportID_AirportName' column: {df['AirportID_AirportName'].isnull().sum()}"

    # null count will artificially show 0 since I assigned values for now
    null_icao = f"Count of null values in 'ICAOCategoryID' column: {df['ICAOCategoryID'].isnull().sum()}"

    with open('outputs/q2_icao_categories/uniqueness_dig.txt', 'w') as file:
        file.write(row_count + '\n')
        file.write(null_air + '\n')
        file.write(null_icao + '\n')
        file.write(combo + '\n')
        file.write(dates + '\n')
        file.write(dates2 + '\n')
        file.write(occid + '\n')
        file.write(occno + '\n')

""" Output some general plots to get going. """
def trend_analysis(df: pd.DataFrame):
    # Plot the monthly trend
    df['Month'] = df['OccDate'].dt.month
    monthly_events = df['Month'].value_counts().sort_index()
    monthly_events.plot(kind='bar')
    plt.title('Monthly Event Trend')
    plt.xlabel('Month')
    plt.ylabel('Number of Events')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.savefig('outputs/q2_icao_categories/event_freq_per_month.png')
    plt.clf()

    # Plot the yearly trend
    df['Year'] = df['OccDate'].dt.year
    yearly_events = df['Year'].value_counts().sort_index()
    yearly_events.plot(kind='line')
    plt.title('Yearly Event Trend')
    plt.xlabel('Year')
    plt.ylabel('Number of Events')
    plt.savefig('outputs/q2_icao_categories/event_freq_per_year.png')
    plt.clf()

    # Pivot to plot trends for each ICAO category
    event_type_trends = df.groupby(['Year', 'ICAO_DisplayEng']).size().reset_index(name='Count')
    event_type_pivot = event_type_trends.pivot(index='Year', columns='ICAO_DisplayEng', values='Count')
    event_type_pivot.plot(kind='line', figsize=(10, 6))
    plt.title('ICAO Category Event Trends Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Events')
    plt.legend(title='ICAO_DisplayEng', fontsize='small', title_fontsize='medium', loc='best',
               bbox_to_anchor=(1.05, 1))
    # plt.tight_layout()
    plt.savefig('outputs/q2_icao_categories/icao_trends.png')
    plt.clf()

def zoom_analysis(df: pd.DataFrame):
    df['Month'] = df['OccDate'].dt.month
    df['Year'] = df['OccDate'].dt.year
    recent_df = df[df['Year'] > 2009]

    # Plot the yearly trend
    yearly_events = recent_df['Year'].value_counts().sort_index()
    yearly_events.plot(kind='line')
    plt.title('Yearly Event Trend Since 2010')
    plt.xlabel('Year')
    plt.ylabel('Number of Events')
    plt.savefig('outputs/q2_icao_categories/event_freq_per_year_gt_2009.png')
    plt.clf()

    # Pivot to plot trends for each ICAO category
    event_type_trends = recent_df.groupby(['Year', 'ICAO_DisplayEng']).size().reset_index(name='Count')
    event_type_pivot = event_type_trends.pivot(index='Year', columns='ICAO_DisplayEng', values='Count')
    event_type_pivot.plot(kind='line', figsize=(10, 6))
    plt.title('ICAO Category Event Trends Over Time Since 2010')
    plt.xlabel('Year')
    plt.ylabel('Number of Events')
    plt.legend(title='ICAO_DisplayEng', fontsize='small', title_fontsize='medium', loc='best',
               bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('outputs/q2_icao_categories/icao_trends_gt_2009.png')
    plt.clf()

    # just 2024
    df_2024 = recent_df[recent_df['Year'] == 2024]
    event_type_trends_2024 = df_2024.groupby(['Month', 'ICAO_DisplayEng']).size().reset_index(name='Count')
    event_type_pivot_2024 = event_type_trends_2024.pivot(index='Month', columns='ICAO_DisplayEng', values='Count')
    event_type_pivot_2024.plot(kind='line', figsize=(10, 6))

    # Get legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Identify the top 10 most common event types based on total occurrences
    top_event_types = df_2024['ICAO_DisplayEng'].value_counts().head(10).index

    # Filter legend handles and labels to include only the top 10
    filtered_handles = [h for h, l in zip(handles, labels) if l in top_event_types]
    filtered_labels = [l for l in labels if l in top_event_types]

    # Customize the legend with filtered handles and labels
    plt.legend(filtered_handles, filtered_labels, title='ICAO_DisplayEng', fontsize='small',
               title_fontsize='medium')
    plt.title('ICAO Category Event Trends Over 2024')
    plt.xlabel('Month')
    plt.ylabel('Number of Events')
    plt.tight_layout()
    plt.savefig('outputs/q2_icao_categories/icao_trends_2024.png')
    plt.clf()

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

    # filter out non-Canadian airports (i.e. ID = 2)
    can_air_df = subset_df[subset_df['AirportID_CountryID'] == 2]

    get_icao_categories(can_air_df)
    get_incident_types(can_air_df)
    explore_data_capture(can_air_df)
    trend_analysis(can_air_df)
    zoom_analysis(can_air_df)



if __name__ == "__main__":
    report_completeness()
    report_icao_categories()

