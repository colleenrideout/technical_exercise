import matplotlib.pyplot as plt
import pandas as pd

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
