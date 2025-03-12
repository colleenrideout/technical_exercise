import matplotlib.pyplot as plt
import pandas as pd

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
