# -*- coding: utf-8 -*-
'''
Program: ExtractData.py
Version: 1
Purpose:  extracts useful data
'''

# Import libraries
import pandas as pd

# Year range from 1992 to 2023
years = range(1992, 2024)
dataframes = []
year_row_counts = []  # Bridge count per year

# Process each year's file
for year in years:
    # File name, e.g. 'CA1992.txt'
    file_name = f'CA{year}.txt'

    print(f'year{year}')
    
    # Read txt as DataFrame, first row as column names
    df = pd.read_csv('./NBIDATA/' + file_name, sep=',', header=0)
    
    # Add 'year' as first column
    df.insert(0, 'year', year)
    
    # Filter to Highway Bridge
    filtered_df = df[df['SERVICE_ON_042A'] == 1]
    # filtered_df = filtered_df[filtered_df['COUNTY_CODE_003'] == 77]
    
    # Select columns with key info (alternative column set)
    #sel_col_df = filtered_df[['year','STRUCTURE_NUMBER_008','SERVICE_LEVEL_005C','COUNTY_CODE_003','ROUTE_NUMBER_005D','FACILITY_CARRIED_007','LAT_016','LONG_017',\
    #                          'YEAR_BUILT_027','ADT_029','HISTORY_037','SERVICE_ON_042A','STRUCTURE_KIND_043A','STRUCTURE_TYPE_043B','STRUCTURE_LEN_MT_049',\
    #                          'DECK_WIDTH_MT_052','DECK_COND_058','SUPERSTRUCTURE_COND_059','SUBSTRUCTURE_COND_060','STRUCTURAL_EVAL_067','WORK_PROPOSED_075A',\
    #                          'TOTAL_IMP_COST_096','YEAR_RECONSTRUCTED_106','PERCENT_ADT_TRUCK_109','NATIONAL_NETWORK_110']]

    # Select columns with key info
    sel_col_df = filtered_df[
        ['year', 'STRUCTURE_NUMBER_008', 'SERVICE_LEVEL_005C', 'COUNTY_CODE_003', 'ROUTE_NUMBER_005D',
         'FACILITY_CARRIED_007', 'KILOPOINT_011','BASE_HWY_NETWORK_012','LAT_016', 'LONG_017',
         'YEAR_BUILT_027', 'ADT_029', 'YEAR_ADT_030','APPR_WIDTH_MT_032',  'HISTORY_037', 'SERVICE_ON_042A', 'STRUCTURE_KIND_043A', 'STRUCTURE_TYPE_043B',
         'MAX_SPAN_LEN_MT_048','STRUCTURE_LEN_MT_049',
         'DECK_WIDTH_MT_052', 'DECK_COND_058', 'SUPERSTRUCTURE_COND_059', 'SUBSTRUCTURE_COND_060', 'CULVERT_COND_062',
         'STRUCTURAL_EVAL_067', 'WORK_PROPOSED_075A', 'IMP_LEN_MT_076',
         'TOTAL_IMP_COST_096', 'YEAR_RECONSTRUCTED_106', 'PERCENT_ADT_TRUCK_109', 'NATIONAL_NETWORK_110']]
    # 49,52,32(new),76(new) relate to funding
    # 58,59,60,62(new) include bridge condition scores

    # Append processed DataFrame to list
    dataframes.append(sel_col_df)

# Concatenate all filtered DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

combined_df.to_excel('output-1992-2023.xlsx', index=False)

#%%
# Bridge count per year
yearly_row_counts = combined_df.groupby('year').size()
yearly_row_counts_array = yearly_row_counts.to_numpy()

# Column for grouping by county
column_to_group = combined_df.iloc[:, 3]
value_counts = column_to_group.value_counts()

#%% Extract raw data
