import pandas as pd

def preprocess_lmp_data(df):
    """
    Preprocess LMP data:
    - Filter for LMP_TYPE == 'LMP'
    - Convert GMT times to America/Los_Angeles timezone (local CA time)
    - Drop timezone info
    - Rename and reorder columns
    - Drop unnecessary columns
    - Sort and add Hour_Label column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data with columns including 'LMP_TYPE', 'INTERVALSTARTTIME_GMT', 'INTERVALENDTIME_GMT', etc.
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame ready for analysis.
    """
    # Filter for LMP_TYPE == 'LMP'
    df = df[df["LMP_TYPE"] == "LMP"].copy()

    # Convert to datetime with UTC
    df['INTERVALSTARTTIME_GMT'] = pd.to_datetime(df['INTERVALSTARTTIME_GMT'], utc=True)
    df['INTERVALENDTIME_GMT'] = pd.to_datetime(df['INTERVALENDTIME_GMT'], utc=True)

    # Convert to California timezone
    df['INTERVALSTARTTIME_CA'] = df['INTERVALSTARTTIME_GMT'].dt.tz_convert('America/Los_Angeles')
    df['INTERVALENDTIME_CA'] = df['INTERVALENDTIME_GMT'].dt.tz_convert('America/Los_Angeles')

    # Drop timezone info
    df['INTERVALSTARTTIME_CA'] = df['INTERVALSTARTTIME_CA'].dt.tz_localize(None)
    df['INTERVALENDTIME_CA'] = df['INTERVALENDTIME_CA'].dt.tz_localize(None)

    # Replace GMT columns with local CA time columns
    df = df.drop(columns=['INTERVALSTARTTIME_GMT', 'INTERVALENDTIME_GMT'])
    df = df.rename(columns={
        'INTERVALSTARTTIME_CA': 'INTERVALSTARTTIME',
        'INTERVALENDTIME_CA': 'INTERVALENDTIME'
    })

    # Drop extra columns that are not needed
    drop_cols = ['NODE_ID_XML', 'OPR_HR', 'NODE_ID', 'PNODE_RESMRID', 'GROUP', 
                 'LMP_TYPE', 'POS', 'MARKET_RUN_ID', 'OPR_INTERVAL', 'XML_DATA_ITEM', 'GRP_TYPE']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Sort by NODE and INTERVALSTARTTIME
    df = df.sort_values(by=['NODE', 'INTERVALSTARTTIME']).reset_index(drop=True)

    # Add Hour_Label column
    df['Hour_Label'] = (
        df['INTERVALSTARTTIME'].dt.strftime('%H:%M') + '-' + 
        (df['INTERVALSTARTTIME'] + pd.Timedelta(hours=1)).dt.strftime('%H:%M')
    )

    return df

def run_preprocessing():
    df5 = pd.read_csv('../data/raw/05.csv')
    df6 = pd.read_csv('../data/raw/06.csv')
    
    df5_preprocessed = preprocess_lmp_data(df5)
    df6_preprocessed = preprocess_lmp_data(df6)
    
    df_combined = pd.concat([df5_preprocessed, df6_preprocessed], ignore_index=True)
    df_combined = df_combined.sort_values(by=['NODE', 'INTERVALSTARTTIME']).reset_index(drop=True)
    
    df_combined.to_csv('../data/preprocessed/0506pre.csv', index=False)
    print("Combined and sorted data saved to 0506pre.csv")

    df_monthly = pd.read_csv('../data/raw/monthly.csv')
    df_monthly_preprocessed = preprocess_lmp_data(df_monthly)
    
    df_monthly_preprocessed.to_csv('../data/preprocessed/monthlypre.csv', index=False)
    print("Monthly preprocessed data saved to monthlypre.csv")

if __name__ == '__main__':
    run_preprocessing()