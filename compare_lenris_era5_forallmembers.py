# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:44:35 2024

@author: Pokhr002
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:13:12 2024

@author: Pokhr002
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
#%% Constants and Directories

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
output_dir = os.path.join(parent_dir, 'output_data')
lentis_dir = "C:\\Users\\pokhr002\\OneDrive - Universiteit Utrecht\\02Data\LENTIS\\"
fig_dir =  os.path.join(parent_dir, 'figure')

START_DATE = '1991-01-01'
END_DATE = '2022-12-31'
LAT_LON_LIMITS = {'lat_min': 28, 'lat_max': 31, 'lon_min': 80.5, 'lon_max': 84}

parameter = 'pr'

# Define a custom theme
custom_theme = {
    'figure.figsize': (10, 6),       # Default figure size
    'axes.labelsize': 12,            # Axis labels font size
    'axes.titlesize': 14,            # Title font size
    'xtick.labelsize': 11,           # X-axis tick labels font size
    'ytick.labelsize': 11,           # Y-axis tick labels font size
    'legend.fontsize': 11,           # Legend font size
    'grid.color': 'grey',            # Grid color
    'grid.linestyle': '--',          # Grid line style
    'grid.linewidth': 0.5,           # Grid line width
    'text.color': 'black',           # Default text color
    'axes.edgecolor': 'black',       # Axis edge color
    'axes.grid': True,               # Enable grid by default
    'axes.spines.top': False,        # Remove top spine
    'axes.spines.right': False,      # Remove right spine
    'savefig.dpi': 300,              # Default save dpi
    'savefig.bbox': 'tight',         # Tight bounding box when saving
    'font.size': 11                  # Default font size for text
}

# Apply the theme
plt.rcParams.update(custom_theme)

#%% Functions

# Function to load dataset
def load_dataset(filepath):
    """Load a NetCDF dataset."""
    try:
        return xr.open_dataset(filepath)
    except FileNotFoundError as e:
        print(f"File not found: {filepath}")
        raise e
        
def convert_lentis_data(df, parameter):
    if parameter == 'pr':
        # Convert precipitation from kg m-2 s-1 to mm/day
        for col in df.columns:
            if col.startswith(f'{parameter}_LENTIS'):
                df[col] = df[col] * 86400  # 1 day = 86400 seconds, so multiply to convert to mm/day
    else:
        # Convert temperature from Kelvin to Celsius
        for col in df.columns:
            if col.startswith(f'{parameter}_LENTIS'):
                df[col] = df[col] - 273.15  # Convert Kelvin to Celsius
    return df        
        
def calculate_rmse_per_group(df_group):
    return np.sqrt(np.mean((df_group[f'{parameter}_LENTIS_members'] - df_group[f'{parameter}_ERA5']) ** 2))
        
# to calculate the bias in dataset
def calculate_percentage_bias(X1, X2):
    return 100 * ((X1 - X2) / X2)

# to calculate the bias in dataset
def calculate_absolute_bias(X1, X2):
    return ((X2 - X1))

# Function to pivot the data based on the index column, lat, lon, and season
def pivot_result_by_lat_lon_season(df, index_col='year', value_col='rmse'):
    df_pivoted = df.pivot_table(
        index=index_col,               # Use the specified index column (e.g., 'year')
        columns=['lat', 'lon', 'season'],  # Pivot based on lat, lon, and season
        values=value_col               # Use the specified value column (e.g., 'rmse')
    )

    # Reorder the MultiIndex columns so that 'lat' and 'lon' are the main level, and 'season' is a sub-level
    df_pivoted = df_pivoted.stack(level=[0, 1]).unstack(level=2)

    # Reset the index to bring the index_col (e.g., 'year') back as a column
    df_pivoted = df_pivoted.reset_index()

    return df_pivoted

#%% 
# This is to get the common lat lon for the LENTIS and KNMI dataset

# Initialize a list to store the DataFrames for all parents
df_all = []

# Iterate over the parent datasets (r01, r02, etc.)
for i in range(1, 17):  # Adjust the range as needed
    parent = f'r{i:02d}'
    print(f"Processing: {parameter}_{parent}_1990-2029_him.nc and {parameter}_ERA5 ")
    
    # Load the dataset for the current parent
    ds1 = load_dataset(os.path.join(lentis_dir, f'{parameter}', f'{parameter}_{parent}_1990-2029_him.nc'))
    ds2 = load_dataset(os.path.join(output_dir, f'{parameter}', f'{parameter}_1991-2022_KB.nc'))
    
    # Check where the variable has non-NaN values in ds2
    has_data = ds2[parameter].notnull()
    
    # Aggregate across the time dimension to find lat/lon points with data at any time point
    has_data_any_time = has_data.any(dim='time')
    
    # Get the latitudes and longitudes where there is data
    non_nan_lats = ds2['lat'].values[has_data_any_time.any(dim='lon')]
    non_nan_lons = ds2['lon'].values[has_data_any_time.any(dim='lat')]
    
    # Initialize a list to store the results for the current parent
    parent_results = []
    
    # Iterate over latitudes and longitudes with data in ds2
    for lat in non_nan_lats:
        for lon in non_nan_lons:
            # Select the nearest grid cell to the specified latitude and longitude in both datasets
            da2_point = ds2[parameter].sel(lat=lat, lon=lon, method='nearest')
            da1_point = ds1[parameter].sel(lat=lat, lon=lon, method='nearest')
            
            # Check if both datasets have values for the given time range
            if da1_point.sel(time=slice(START_DATE, END_DATE)).notnull().all() and da2_point.sel(time=slice(START_DATE, END_DATE)).notnull().all():
                # Select the data for the specified date range
                da1_selected = da1_point.sel(time=slice(START_DATE, END_DATE))
                da2_selected = da2_point.sel(time=slice(START_DATE, END_DATE))
                
                # Convert the selected data to a DataFrame
                df = pd.DataFrame({
                    'time': da1_selected['time'].values,
                    'lat': lat,
                    'lon': lon,
                    f'{parameter}_LENTIS_{parent}': da1_selected.values,
                    f'{parameter}_ERA5': da2_selected.values
                })
                
                # Append the DataFrame to the results list for this parent
                parent_results.append(df)
    
    # Concatenate all DataFrames for the current parent into one DataFrame
    if parent_results:
        df_parent = pd.concat(parent_results, ignore_index=True)
        # Append the parent's DataFrame to the overall list
        df_all.append(df_parent)

# If you want to combine all parents into a single DataFrame after the loop
df_combined = df_all[0]
for df in df_all[1:]:
    df_combined = pd.merge(df_combined, df, on=['time', 'lat', 'lon', f'{parameter}_ERA5'], how='outer')
print(df_combined)

df_combined = convert_lentis_data(df_combined, parameter)

# Define year and the seasons for the df_combined
df_combined['year'] = df_combined['time'].dt.year
pre_monsoon_months = [3, 4, 5]  # March to May
monsoon_months = [6, 7, 8, 9]   # June to September
post_monsoon_months = [10, 11]  # October to November
winter_months = [12, 1, 2]      # December to February
df_combined['month'] = df_combined['time'].dt.month

# Assign seasons based on the month
conditions = [
    df_combined['month'].isin(pre_monsoon_months),
    df_combined['month'].isin(monsoon_months),
    df_combined['month'].isin(post_monsoon_months),
    df_combined['month'].isin(winter_months)
]

# Adding seasons to the df
choices = ['pre-monsoon', 'monsoon', 'post-monsoon', 'winter']
df_combined['season'] = np.select(conditions, choices, default='unknown')

average_columns = [col for col in df_combined.columns if col.startswith(f'{parameter}_LENTIS')]
# Create a new column to store the average of all the members
df_combined[f'{parameter}_LENTIS_members'] = df_combined[average_columns].mean(axis=1)
cols = list(df_combined.columns)  # Get the current list of columns
cols.insert(3, cols.pop(cols.index(f'{parameter}_LENTIS_members')))  # Move the average column to the 4th position
df_combined = df_combined[cols]

# df_numeric = df_combined.select_dtypes(include=['number']) # Create a boolean mask for values <= -1
# mask = (df_numeric <= -1) # Create a boolean mask for values <= -1
# rows_all_negative_or_equal = df_combined[mask.all(axis=1)] # Find rows where all numeric columns satisfy the condition
# print(rows_all_negative_or_equal) # Display the resulting DataFrame


# Display the resulting DataFrame
print(df_combined)
output_file = os.path.join(output_dir, f'{parameter}_df_combined.csv')
df_combined.to_csv(output_file, index=False)

# Print confirmation
print(f"DataFrame saved to: {output_file}")

selected_columns = ['time', 'year', 'month','season', 'lat', 'lon', f'{parameter}_ERA5',f'{parameter}_LENTIS_members']
df_LENTIS_ERA5 = df_combined[selected_columns]
print(df_LENTIS_ERA5)

#%%

# THIS CODE IS TO CALCULATE THE STATISTICS FOR LONG TERM DATA AND SEASONS IN YEARS for all the members

def flatten_columns(df):
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
    
aggregations = {}   
for col in df_combined.columns:
    if col.startswith(f'{parameter}_LENTIS') or col.startswith(f'{parameter}_ERA5'):
        if parameter == 'pr':
            aggregations[col] = ['sum']  # Use 'sum' for precipitation
        else:
            aggregations[col] = ['mean']  # Use 'mean' for other parameters

        
# df_long_term = df_combined.groupby(['season', 'lat', 'lon']).agg(aggregations).reset_index()
df_yearly = df_combined.groupby(['year', 'lat', 'lon']).agg(aggregations).reset_index()
df_yearly_month = df_combined.groupby(['lat', 'lon','year','month']).agg(aggregations).reset_index()
df_yearly_season = df_combined.groupby(['year', 'season', 'lat', 'lon']).agg(aggregations).reset_index()
flatten_columns(df_yearly_month) #flattening the columns with two titles
flatten_columns(df_yearly_season)
flatten_columns(df_yearly)
df_yearly.rename(columns={'lat_': 'lat', 'lon_': 'lon', 'year_': 'year'}, inplace=True)
df_yearly_month.rename(columns={'lat_': 'lat', 'lon_': 'lon', 'year_': 'year', 'month_': 'month'}, inplace=True)
df_yearly_season.rename(columns={'lat_': 'lat', 'lon_': 'lon', 'year_': 'year', 'season_': 'season'}, inplace=True)


std_var_aggregations = {}
for col in df_yearly.columns:
    if col not in ['year', 'lat', 'lon', 'month', 'season']:  # Exclude non-numeric columns
        std_var_aggregations[col] = ['mean', 'std', 'var']
        

df_long_term_monthly_mean = df_yearly_month.groupby(['month', 'lat', 'lon']).agg(std_var_aggregations).reset_index()
df_long_term_seasonal_mean = df_yearly_season.groupby(['season', 'lat', 'lon']).agg(std_var_aggregations).reset_index()
df_yearly_overall_mean = df_yearly.groupby(['year']).agg(std_var_aggregations).reset_index()

flatten_columns(df_long_term_monthly_mean)
flatten_columns(df_long_term_seasonal_mean)
flatten_columns(df_yearly_overall_mean)


df_long_term_monthly_mean.rename(columns={'lat_': 'lat', 'lon_': 'lon', 'month_': 'month'}, inplace=True)
df_long_term_seasonal_mean.rename(columns={'lat_': 'lat', 'lon_': 'lon', 'season_': 'season'}, inplace=True)
df_yearly_overall_mean.rename(columns={'lat_': 'lat', 'lon_': 'lon', 'year_': 'year'}, inplace=True)

print(df_long_term_monthly_mean.head())
print(df_yearly_season.head())
print(df_long_term_seasonal_mean.head())
print(df_yearly.head())


# # Calculate bias between LENTIS members and ERA5
if parameter == 'pr':
    df_yearly_season['%bias'] = calculate_percentage_bias(df_yearly_season[f'{parameter}_LENTIS_members_sum'], df_yearly_season[f'{parameter}_ERA5_sum'])
    df_long_term_seasonal_mean['%bias'] = calculate_percentage_bias(df_long_term_seasonal_mean[f'{parameter}_LENTIS_members_sum_mean'], df_long_term_seasonal_mean[f'{parameter}_ERA5_sum_mean'])
else:
    df_yearly_season['abs_bias'] = calculate_absolute_bias(df_yearly_season[f'{parameter}_LENTIS_members_mean'], df_yearly_season[f'{parameter}_ERA5_mean'])
    df_long_term_seasonal_mean['abs_bias'] = calculate_absolute_bias(df_long_term_seasonal_mean[f'{parameter}_LENTIS_members_mean_mean'], df_long_term_seasonal_mean[f'{parameter}_ERA5_mean_mean'])



# # Calculate RMSE
# df_long_term_rmse = df_LENTIS_ERA5.groupby(['season', 'lat', 'lon']).apply(calculate_rmse_per_group).reset_index(name='rmse')
# df_yearly_rmse = df_LENTIS_ERA5.groupby(['year', 'lat', 'lon']).apply(calculate_rmse_per_group).reset_index(name='rmse')  # Calculate RMSE for yearly
# df_yearly_season_rmse = df_LENTIS_ERA5.groupby(['year', 'season', 'lat', 'lon']).apply(calculate_rmse_per_group).reset_index(name='rmse')  # Calculate RMSE for yearly seasonal

# # Merge RMSE with the aggregated DataFrames
# df_long_term = pd.merge(df_long_term, df_long_term_rmse, on=['season', 'lat', 'lon'], how='left')
# df_yearly = pd.merge(df_yearly, df_yearly_rmse, on=['year', 'lat', 'lon'], how='left')
# df_yearly_season = pd.merge(df_yearly_season, df_yearly_season_rmse, on=['year', 'season', 'lat', 'lon'], how='left')

# # Display the resulting DataFrames
# print(df_long_term)
# print(df_yearly)
# print(df_yearly_season)

#%%
##### THIS IS THE PLOT OF LONG TERM MONTHLY MEAN OF THE PARAMETER ACROSS GRID AND COMPARISON WITH LENTIS AND ERA5
### 01

plot_type = 'lineplot_monthly'
if parameter == 'pr':
    unit = "mm/month"
else:
    unit = "°C"

def plot_line_plots_subplots(df, parameter, rows=4, cols=5):
    # Define color-blind-friendly colors
    colors = {
        'LENTIS': 'black',  
        'ERA5': 'royalblue',    
        'Shaded Area': '#D55E00'  
    }

    # Month labels as "J", "F", "M", etc.
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    # Extract the appropriate columns based on the parameter
    if parameter == 'pr':
        lentis_col_suffix = '_sum_mean'
        era5_col_suffix = '_sum_mean'
    else:
        lentis_col_suffix = '_mean_mean'
        era5_col_suffix = '_mean_mean'
        
    lentis_columns = [col for col in df.columns if col.startswith(f'{parameter}_LENTIS_r') and col.endswith(lentis_col_suffix)]

    # Get unique lat-lon combinations and sort them by lat and lon in descending order
    unique_locs = df_long_term_monthly_mean[['lat', 'lon']].drop_duplicates().sort_values(by=['lat', 'lon'], ascending=[False, True])

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15)) #, sharey=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Loop over each unique lat-lon pair and create a subplot
    for i, (lat, lon) in enumerate(unique_locs.itertuples(index=False)):
        if i >= rows * cols:
            break  # Stop if there are more grid cells than available subplots

        # Filter the DataFrame for the current lat-lon combination
        df_loc = df[(df['lat'] == lat) & (df['lon'] == lon)]

        # Extract the relevant data for plotting
        months = df_loc['month']
        pr_lentis_members = df_loc[f'{parameter}_LENTIS_members{lentis_col_suffix}']
        pr_era5 = df_loc[f'{parameter}_ERA5{era5_col_suffix}']

        # Calculate the shaded area (min and max across individual LENTIS members)
        pr_lentis_min = df_loc[lentis_columns].min(axis=1)
        pr_lentis_max = df_loc[lentis_columns].max(axis=1)

        ax = axes[i]

        # Shaded area
        ax.fill_between(months, pr_lentis_min, pr_lentis_max, color=colors['Shaded Area'], alpha=0.6, label='LENTIS Members Range')

        # Plot the lines for LENTIS members sum and ERA5 sum
        ax.plot(months, pr_lentis_members, label=f'{parameter}_LENTIS_members monthly value', color=colors['LENTIS'])
        ax.plot(months, pr_era5, label=f'{parameter}_ERA5 monthly value', color=colors['ERA5'])

        # Plot the mean value lines for LENTIS members sum and ERA5 sum
        ax.axhline(y=np.mean(pr_lentis_members), color=colors['LENTIS'], linestyle='--', linewidth=1.5, label='LENTIS mean line')
        ax.axhline(y=np.mean(pr_era5), color=colors['ERA5'], linestyle='--', linewidth=1.5, label='ERA5 mean line')

        # Adding labels and title
        ax.set_xticks(ticks=np.arange(1, 13))
        ax.set_xticklabels(month_labels)  # Set the x-axis with month abbreviations
        ax.set_title(f'Lat: {round(lat, 2)}, Lon: {round(lon, 2)}')
        if i % cols == 0:
            ax.set_ylabel(f'{parameter} ({unit})')
        else:
            ax.set_ylabel('')

        ax.legend().set_visible(False)  # Hide legend on individual subplots to avoid clutter

    # Hide any remaining empty subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    # Adjust layout and add a single legend
    plt.suptitle(f'Monthly {parameter.capitalize()} Comparison: LENTIS Members vs ERA5 across grids', fontsize=16, color="red", weight='bold', backgroundcolor='lightgray')
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2)

    # Save the plot
    os.makedirs(new_dir := os.path.join(fig_dir, parameter, f'01_{plot_type}'), exist_ok=True)
    plot_filename = os.path.join(new_dir, f'{parameter}_{plot_type}.png')
    print(f"Saving plot to: {plot_filename}") 
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

# Calling the function
plot_line_plots_subplots(df_long_term_monthly_mean, parameter, rows=4, cols=5)

#%%

## This is to plot the changes in year over the grids between LENTIS and ERA5

plot_type = 'lineplot_yearly'
if parameter == 'pr':
    unit = "mm/year"
else:
    unit = "°C"

def plot_yearly_line_plots_subplots(df, parameter, rows=4, cols=5):
    # Define color-blind-friendly colors
    colors = {
        'LENTIS': 'black',  
        'ERA5': 'royalblue',    
        'Shaded Area': '#D55E00'  
    }

    # Extract the appropriate columns based on the parameter
    if parameter == 'pr':
        lentis_col_suffix = '_sum'
        era5_col_suffix = '_sum'
    else:
        lentis_col_suffix = '_mean'
        era5_col_suffix = '_mean'
        
    lentis_columns = [col for col in df.columns if col.startswith(f'{parameter}_LENTIS_r') and col.endswith(lentis_col_suffix)]

    # Get unique lat-lon combinations and sort them by lat and lon in descending order
    unique_locs = df[['lat', 'lon']].drop_duplicates().sort_values(by=['lat', 'lon'], ascending=[False, True])

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15) )#, sharey=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Loop over each unique lat-lon pair and create a subplot
    for i, (lat, lon) in enumerate(unique_locs.itertuples(index=False)):
        if i >= rows * cols:
            break  # Stop if there are more grid cells than available subplots

        # Filter the DataFrame for the current lat-lon combination
        df_loc = df[(df['lat'] == lat) & (df['lon'] == lon)]

        # Extract the relevant data for plotting
        years = df_loc['year']
        pr_lentis_members = df_loc[f'{parameter}_LENTIS_members{lentis_col_suffix}']
        pr_era5 = df_loc[f'{parameter}_ERA5{era5_col_suffix}']

        # Calculate the shaded area (min and max across individual LENTIS members)
        pr_lentis_min = df_loc[lentis_columns].min(axis=1)
        pr_lentis_max = df_loc[lentis_columns].max(axis=1)

        ax = axes[i]

        # Shaded area
        ax.fill_between(years, pr_lentis_min, pr_lentis_max, color=colors['Shaded Area'], alpha=0.5, label='LENTIS Members Range')

        # Plot the lines for LENTIS members sum and ERA5 sum
        ax.plot(years, pr_lentis_members, label=f'{parameter}_LENTIS_members yearly value', color=colors['LENTIS'])
        ax.plot(years, pr_era5, label=f'{parameter}_ERA5 yearly value', color=colors['ERA5'])

        # Plot the mean value lines for LENTIS members sum and ERA5 sum
        ax.axhline(y=np.mean(pr_lentis_members), color=colors['LENTIS'], linestyle='--', linewidth=1.5, label='LENTIS mean line')
        ax.axhline(y=np.mean(pr_era5), color=colors['ERA5'], linestyle='--', linewidth=1.5, label='ERA5 mean line')

        # Adding labels and title
        ax.set_xticks(ticks=np.arange(years.min(), years.max() + 1, 2))
        ax.set_xticklabels(years[::2], rotation=45)  # Set the x-axis with years
        ax.set_title(f'Lat: {round(lat, 2)}, Lon: {round(lon, 2)}')
        if i % cols == 0:
            ax.set_ylabel(f'{parameter} {unit}')
        else:
            ax.set_ylabel('')

        ax.legend().set_visible(False)  # Hide legend on individual subplots to avoid clutter

    # Hide any remaining empty subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    # Adjust layout and add a single legend
    plt.suptitle(f'Yearly_{parameter.capitalize()} Comparison: LENTIS Members vs ERA5 across grids', fontsize=16, color="red", weight='bold', backgroundcolor='lightgray')
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2)

    # Save the plot
    os.makedirs(new_dir := os.path.join(fig_dir, parameter, f'02_{plot_type}'), exist_ok=True)
    plot_filename = os.path.join(new_dir, f'{parameter}_{plot_type}.png')
    print(f"Saving plot to: {plot_filename}") 
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

# Calling the function
plot_yearly_line_plots_subplots(df_yearly, parameter, rows=4, cols=5)

#%%
#### PLOTTING THE BAR PLOTS ACROSS GRIDS FOR LENTIS AND ERA5 

############ THIS IS CALCULATION OF MEAN VALUE OF SEASONS FOR LENTIS MEMBER MEAN AND ERA5

plot_type = 'barplot_season'

if parameter == 'pr':
    unit = "mm/season"
else:
    unit = "°C"

def plot_seasonal_bar_plots_subplots(df, parameter, rows=4, cols=5):
    # Extract the appropriate columns based on the parameter
    if parameter == 'pr':
        lentis_col_suffix = '_sum_mean'
        era5_col_suffix = '_sum_mean'
    else:
        lentis_col_suffix = '_mean_mean'
        era5_col_suffix = '_mean_mean'
        
    # Get the relevant columns for LENTIS members and ERA5
    lentis_col = f'{parameter}_LENTIS_members{lentis_col_suffix}'
    era5_col = f'{parameter}_ERA5{era5_col_suffix}'
    if parameter == 'pr':
       bias_col = '%bias'
    else:
       bias_col = 'abs_bias'
    

    # Get unique lat-lon combinations and sort them by lat and lon in descending order
    unique_locs = df[['lat', 'lon']].drop_duplicates().sort_values(by=['lat', 'lon'], ascending=[False, True])

    # Limit the color palette to the number of unique hue values (seasons)
    num_hue_levels = df['season'].nunique()
    colors = sns.color_palette("Paired", num_hue_levels)

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15), sharey=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    bar_width = 0.35  # Width of the bars

    # Loop over each unique lat-lon pair and create a subplot
    for i, (lat, lon) in enumerate(unique_locs.itertuples(index=False)):
        if i >= rows * cols:
            break  # Stop if there are more grid cells than available subplots

        # Filter the DataFrame for the current lat-lon combination
        df_loc = df[(df['lat'] == lat) & (df['lon'] == lon)]

        ax = axes[i]

        # Create an index for each season's position on the x-axis
        season_positions = np.arange(len(df_loc['season'].unique()))
        
        # Plot the bars for LENTIS members and ERA5
        bars_lentis = ax.bar(season_positions - bar_width/2, df_loc[lentis_col], bar_width, label='LENTIS_all members', color=colors[0])
        ax.bar(season_positions + bar_width/2, df_loc[era5_col], bar_width, label='ERA5', color=colors[2])

        # Adding bias percentage as text on top of each bar
        for bar_lentis, bias in zip(bars_lentis, df_loc[bias_col]):
            height_lentis = bar_lentis.get_height()            
            ax.text(bar_lentis.get_x() + bar_lentis.get_width()/2., height_lentis + 0.05, f'{bias:.1f}%', ha='center', va='bottom', fontsize=10)
            
        # Adding labels and title
        ax.set_xticks(season_positions)
        ax.set_xticklabels(df_loc['season'].unique(), rotation=45)
        ax.set_title(f'Lat: {round(lat, 2)}, Lon: {round(lon, 2)}')
        if i % cols == 0:
            ax.set_ylabel(f'{parameter} {unit}')
        else:
            ax.set_ylabel('')

        ax.legend().set_visible(False)  # Hide legend on individual subplots to avoid clutter

    # Hide any remaining empty subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    # Adjust layout and add a single legend
    plt.suptitle(f'Seasonal {parameter.capitalize()} Comparison: LENTIS Members vs ERA5 across grids', fontsize=16, color="red", weight='bold', backgroundcolor='lightgray')
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2)

    # Save the plot
    os.makedirs(new_dir := os.path.join(fig_dir, parameter, f'03_{plot_type}'), exist_ok=True)
    plot_filename = os.path.join(new_dir, f'{parameter}_{plot_type}.png')
    print(f"Saving plot to: {plot_filename}") 
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

# Calling the function
plot_seasonal_bar_plots_subplots(df_long_term_seasonal_mean, parameter, rows=4, cols=5)

#%%
plot_type = 'barplot_season'

if parameter == 'pr':
    unit = "mm/season"
else:
    unit = "°C"

def plot_seasonal_bar_plots_subplots(df, parameter, rows=4, cols=5):
    # Filter the DataFrame for the monsoon and winter seasons
    df_filtered = df[df['season'].isin(['monsoon', 'winter'])]
   
    # Extract the appropriate columns based on the parameter
    if parameter == 'pr':
        lentis_col_suffix = '_sum_mean'
        era5_col_suffix = '_sum_mean'
    else:
        lentis_col_suffix = '_mean_mean'
        era5_col_suffix = '_mean_mean'
            
    # Get the relevant columns for LENTIS members
    lentis_columns = [col for col in df_filtered.columns if col.startswith(f'{parameter}_LENTIS_r') and col.endswith(lentis_col_suffix)]
    era5_columns = [col for col in df_filtered.columns if col.startswith(f'{parameter}_ERA5') and col.endswith(era5_col_suffix)]

    # Loop through each season to create separate plots
    for season in ['monsoon', 'winter']:
        # Filter the DataFrame for the current season
        df_season = df_filtered[df_filtered['season'] == season]

        # Get unique lat-lon combinations and sort them by lat and lon in descending order
        unique_locs = df_season[['lat', 'lon']].drop_duplicates().sort_values(by=['lat', 'lon'], ascending=[False, True])

        # Limit the color palette to the number of unique hue values (seasons)
        color_palette = sns.color_palette("Paired")

        # Create a figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15), sharey=True)
        axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

        bar_width = 0.35  # Width of the bars

        # Loop over each unique lat-lon pair and create a subplot
        for i, (lat, lon) in enumerate(unique_locs.itertuples(index=False)):
            if i >= rows * cols:
                break  # Stop if there are more grid cells than available subplots

            # Filter the DataFrame for the current lat-lon combination
            df_loc = df_season[(df_season['lat'] == lat) & (df_season['lon'] == lon)]
            print(df_loc)

            ax = axes[i]
            
            # Get the values for the LENTIS and ERA5 columns
            lentis_values = df_loc[lentis_columns].values.flatten()
            print('The lentis values are:',lentis_values)
            
            era5_values = df_loc[era5_columns].values.flatten()
            print('The ERA5 members are:', era5_values)
            
            # Create an index for each LENTIS column's position on the x-axis
            bar_positions = np.arange(len(lentis_columns))

            # Plot the bars for LENTIS members
            ax.bar(bar_positions - bar_width/2, lentis_values, bar_width, color= color_palette[0], label='LENTIS Members')
            ax.axhline(y=era5_values, color='red', linestyle='--', linewidth=2, label='ERA5')
            # ax.text(len(lentis_columns), era5_values, f'ERA5: {era5_value:.2f}', color=color_palette[2], va='center', ha='right', fontsize=10)

            # Adding labels and title
            ax.set_xticks(bar_positions)
            ax.set_xticklabels([f'LENTIS {j+1}' for j in range(len(lentis_columns))], rotation=90)
            ax.set_title(f'Lat: {round(lat, 2)}, Lon: {round(lon, 2)}')
            if i % cols == 0:
                ax.set_ylabel(f'{parameter} {unit}')
            else:
                ax.set_ylabel('')

            ax.legend().set_visible(False)  # Hide legend on individual subplots to avoid clutter

        # Hide any remaining empty subplots
        for j in range(i + 1, rows * cols):
            fig.delaxes(axes[j])

        # Adjust layout and add a single legend
        plt.suptitle(f'Seasonal {parameter.capitalize()} Comparison for {season.capitalize()}: LENTIS Members vs ERA5 across grids', fontsize=16, color="red", weight='bold', backgroundcolor='lightgray')
        plt.tight_layout()
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=2)

        # Save the plot
        os.makedirs(new_dir := os.path.join(fig_dir, parameter, f'03_{plot_type}'), exist_ok=True)
        plot_filename = os.path.join(new_dir, f'{parameter}_{season}_{plot_type}.png')
        print(f"Saving plot to: {plot_filename}") 
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()

# Calling the function
plot_seasonal_bar_plots_subplots(df_long_term_seasonal_mean, parameter, rows=4, cols=5)


#%%

plot_type = 'histogram_spatial'

def plot_spatial_histograms_by_lat_lon(df_filtered, parameter, rows=4, cols=5, suffix='xx', title='yy', unit='zz'):
    
    # Get the list of columns related to LENTIS members based on the suffix
    lentis_columns = [col for col in df_filtered.columns if col.startswith(f'{parameter}_LENTIS_r') and col.endswith(suffix)]

    # Get unique lat-lon combinations
    unique_locs = df_filtered[['lat', 'lon']].drop_duplicates().sort_values(by=['lat', 'lon'], ascending=[False, True])

    # Loop through the seasons to create separate plots
    for season in ['monsoon', 'winter']:
        # Create subplots for each season
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

        for i, (lat, lon) in enumerate(unique_locs.itertuples(index=False)):
            if i >= len(axes):
                break  # Stop if there are more grid cells than available subplots

            ax = axes[i]
            print(f"Plotting for Lat: {lat}, Lon: {lon}")

            # Filter the data for the current lat-lon combination and season
            df_loc = df_filtered[(df_filtered['lat'] == lat) & (df_filtered['lon'] == lon) & (df_filtered['season'] == season)]

            # Combine all the LENTIS member columns into a single series for plotting
            print(f"Columns being used: {lentis_columns}")            
            values = df_loc[lentis_columns].values.flatten()
            lentis_mean = float(values.mean())
            print(f"The {suffix.lower()} values are:\n", values)    

            max_val = np.ceil(values.max()).astype(int)
            min_val = np.floor(values.min())
            bin_width = 5
                                    
            if suffix == '_std' and parameter == 'pr':           
                # Calculate the bin edges with equal distribution and fixed width
                bin_edges = np.arange(min_val, max_val + bin_width, bin_width)
            else:            
                # Calculate the bin edges with equal distribution and fixed width
                bin_edges = np.arange(min_val, max_val + 1, 1)
            
            # Plot the histogram
            sns.histplot(values, kde=False, color='skyblue', bins=bin_edges, ax=ax, edgecolor='black', linewidth=0.8, label='Value of each lentis members')
  
            
            # Plot for the ERA5 as well, only for the _std suffix
            era5_value = df_loc[[col for col in df_loc.columns if col.startswith(f'{parameter}_ERA5') and col.endswith(suffix)]].values[0] 
            era5_value = float(era5_value)
            ax.axvline(era5_value, color='red', linestyle='--', linewidth=2, label='ERA5')
            ax.axvline(lentis_mean, color='blue', linestyle='-', linewidth=2, label='LENTIS')
            ax.text(era5_value, ax.get_ylim()[1] * 1.2, f'{round(era5_value, 2)}', color='red', ha='center', va='bottom', fontsize=10, rotation=270)
            ax.text(lentis_mean, ax.get_ylim()[1] * 1.2, f'{round(lentis_mean, 2)}', color='blue', ha='center', va='bottom', fontsize=10, rotation=270)

            print("The value of the era5 data:", era5_value,"\n")
            
            # Set labels and title
            ax.set_title(f'Lat: {round(lat, 2)}, Lon: {round(lon, 2)}')
            # ax.set_xticklabels(np.arange(min_val, max_val, step=5), rotation=45)
            ax.set_xlabel(f'{parameter}_{title}({unit})')
            ax.set_ylabel('Frequency')
            ax.legend().set_visible(False)
            
        # Hide any remaining empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and show the plot
        plt.suptitle(f'Histograms of {title}_{parameter} for LENTIS Members during {season.capitalize()} across Grids', fontsize=16, color="red", weight='bold', backgroundcolor='lightgray')
        plt.tight_layout()
        os.makedirs(new_dir := os.path.join(fig_dir, parameter, f'04_{plot_type}', season), exist_ok=True)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=2)
        
        plot_filename = os.path.join(new_dir, f'{parameter}_{season}_{suffix}_histogram_spatial.png')
        print(f"Saving plot to: {plot_filename}") 
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()


# Filter the DataFrame for the monsoon and winter seasons
df_filtered = df_long_term_seasonal_mean[df_long_term_seasonal_mean['season'].isin(['monsoon', 'winter'])]

# Plot for standard deviation
plot_spatial_histograms_by_lat_lon(df_filtered, parameter, rows=4, cols=5, suffix='_std', title='Standard Deviation', unit='-')

# Plot for sum
if parameter == 'pr':
    plot_spatial_histograms_by_lat_lon(df_filtered, parameter, rows=4, cols=5, suffix='_mean',title='Average', unit='mm/month')
else:
    plot_spatial_histograms_by_lat_lon(df_filtered, parameter, rows=4, cols=5, suffix='_mean',title='Average', unit='°C')
    
#%%

plot_type = 'histogram_stddev_allgrids'

def plot_histograms(df_filtered, parameter, rows=4, cols=5):
    
    lentis_columns = [col for col in df_filtered.columns if col.startswith(f'{parameter}_LENTIS_r') and col.endswith('_std')]

    # Loop through the seasons to create separate plots
    for season in ['monsoon', 'winter']:
        # Filter the data for the current season
        df_season = df_filtered[df_filtered['season'] == season]
        print (df_season)

        # Combine all the standard deviation columns into a single series for plotting
        print(f"Columns being used for standard deviations: {lentis_columns}")            
        std_values = df_season[lentis_columns].values.flatten()
        print("The std dev values are:", std_values)    

        # Calculate the bin edges based on the max standard deviation
        max_std = np.ceil(std_values.max()).astype(int)
        bin_edges = np.arange(0, max_std + 1, 1)

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(std_values, kde=False, color='skyblue', bins=bin_edges)

        # Plot for the ERA5 as well (mean of all lat-lon combinations)
        era5_std_column = df_season[[col for col in df_season.columns if col.startswith(f'{parameter}_ERA5') and col.endswith('_std')]]
        era5_std = era5_std_column.mean().iloc[0]
        print (era5_std)
        plt.axvline(era5_std, color='red', linestyle='--', linewidth=2, label=f'ERA5 Std Dev: {round(era5_std, 2)}')
                         
        # Set labels and title
        plt.title(f'Histograms of Standard Deviations for LENTIS Members during {season.capitalize()}', fontsize=16, color="red")
        plt.xlabel(f'Standard Deviation ({parameter})')
        plt.ylabel('Frequency')
        plt.legend()

        # Save the plot
        os.makedirs(new_dir := os.path.join(fig_dir, parameter, f'05_{plot_type}', season), exist_ok=True)
        plot_filename = os.path.join(new_dir, f'{parameter}_{season}_stddev_histogram.png')
        print(f"Saving plot to: {plot_filename}") 
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()

# Filter the DataFrame for the monsoon and winter seasons
df_filtered = df_long_term_seasonal_mean[df_long_term_seasonal_mean['season'].isin(['monsoon', 'winter'])]
plot_histograms(df_filtered, parameter, rows=4, cols=5)    


#%%
# OVERALL HISTOGRAM PLOT OF THE PRECIPITATION IN SUMMER AND WINTER OVERALL

plot_type = 'histogram_seasons_dailydata_allgrids'
if parameter == 'pr':
    unit = "mm/day"
else:
    unit = "°C"
    
df_filtered = df_combined[df_combined['season'].isin(['monsoon', 'winter'])]


# Define the width of the bars
bar_width = 2.0

# Initialize a dictionary to store histogram data
histogram_data = {'season': [], 'bin_centers': [], f'{parameter}_LENTIS_members': [], f'{parameter}_ERA5': []}

# Loop through the seasons to create histograms
seasons = ['monsoon', 'winter']

for season in seasons:
    plt.figure(figsize=(16, 8))  # Create a new figure for each season
    colors = sns.color_palette("Paired") 
       
    # Filter the data for the current season
    df_season = df_filtered[df_filtered['season'] == season]
    
    # Aggregate all LENTIS members' data into a single array
    lentis_columns = [col for col in df_season.columns if col.startswith(f'{parameter}_LENTIS_r')]
    lentis_values = df_season[lentis_columns].values.flatten()
    
    # Define bin edges with 5 mm intervals (adjust to match your parameter's unit)
    bin_edges = np.arange(0, df_season[[f'{parameter}_ERA5']].max().max() + 5, 5)

    # Define the bin centers for positioning the bars
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Calculate histograms for the season
    hist_data_LENTIS_members, _ = np.histogram(lentis_values, bins=bin_edges)
    hist_data_LENTIS_members = np.ceil(hist_data_LENTIS_members / 16).astype(int)  # Adjust the frequency by dividing by 16
    hist_data_ERA5, _ = np.histogram(df_season[f'{parameter}_ERA5'], bins=bin_edges)

    # Save histogram data
    histogram_data['season'].extend([season] * len(bin_centers))
    histogram_data['bin_centers'].extend(bin_centers)
    histogram_data[f'{parameter}_LENTIS_members'].extend(hist_data_LENTIS_members)
    histogram_data[f'{parameter}_ERA5'].extend(hist_data_ERA5)

    # Plot side-by-side bar plots
    plt.bar(bin_centers - bar_width/2, hist_data_LENTIS_members, width=bar_width, edgecolor='black', label=f'{parameter}_LENTIS_members-{season}', color=colors[6], alpha=0.8)
    plt.bar(bin_centers + bar_width/2, hist_data_ERA5, width=bar_width, edgecolor='black', label=f'{parameter} ERA5-{season}', color=colors[0], alpha=0.8)

    # Add frequency values on top of the bars
    for i, (count1, count2) in enumerate(zip(hist_data_LENTIS_members, hist_data_ERA5)):
        plt.text(bin_centers[i] - bar_width/2, count1 + 0.1, str(round(count1, 2)), ha='center', va='bottom', fontsize=9, rotation=270)
        plt.text(bin_centers[i] + bar_width/2, count2 + 0.1, str(count2), ha='center', va='bottom', fontsize=9, rotation=270)

    # Option 1: Use a log scale to manage large differences
    plt.yscale('log')

    # Set labels and title
    plt.xlabel(f'{parameter.capitalize()} ({unit})')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of daily {parameter.capitalize()} from all grids for {season}')
    plt.tight_layout()

    # Add more x-ticks
    max_x = bin_edges[-1]
    plt.xticks(bin_edges, rotation=45)

    plt.legend()
    plt.tight_layout()

    # Save the plot for the current statistic
    os.makedirs(new_dir := os.path.join(fig_dir, parameter, f'06_{plot_type}'), exist_ok=True)
    plot_filename = os.path.join(new_dir, f'Histogram_allgrids_{parameter}_{season}.png')
    print(f"Saving plot to: {plot_filename}") 
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()  
    plt.close()   


#%%

plot_type = 'barplot_bias'

def plot_multiple_stats_barplots(df, stats, rows=4, cols=5):
    
    # Filter the data for monsoon and winter seasons only
    df_filtered = df[df['season'].isin(['monsoon', 'winter'])]
    print(df_filtered.head())  # Check the first few rows of the filtered data
    
    # Get unique lat-lon combinations
    unique_locs = df_filtered[['lat', 'lon']].drop_duplicates().sort_values(by=['lat', 'lon'], ascending=[False, True])

    # Loop over each statistic to create separate plots
    for stat in stats:
        # Create a figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

        # Limit the color palette to the number of unique hue values
        color_palette = sns.color_palette("Paired")
        selected_colors = [color_palette[1], color_palette[2]] 

        # Loop over each unique lat-lon pair and create a subplot
        for i, (lat, lon) in enumerate(unique_locs.itertuples(index=False)):
            if i >= rows * cols:
                break  # Stop if there are more grid cells than available subplots

            # Filter the data for the current lat-lon combination
            df_loc = df_filtered[(df_filtered['lat'] == lat) & (df_filtered['lon'] == lon)]
            
            # If plotting %bias, filter out values greater than 100
            # if stat == '%bias':
            #    df_loc = df_loc[df_loc[stat] <= 100]
               
            # Check if DataFrame is empty before plotting
            if df_loc.empty:
                print(f"No data for lat: {lat}, lon: {lon}")
                continue
            
            print(df_loc.head())

            ax = axes[i]

            # Plot bar plots for the given stat
            sns.barplot(
                data=df_loc,
                x='year', 
                y=stat, 
                hue='season',
                ax=ax, 
                palette=selected_colors,  # Use the limited color palette
                errorbar=None
            )

            # Set title and labels
            ax.set_title(f'Lon: {round(lon, 2)}, Lat: {round(lat, 2)}')
            years_sorted = np.sort(df_loc['year'].unique())  
            ax.set_xticks(np.arange(0, len(years_sorted), 2)) 
            ax.set_xticklabels(years_sorted[::2], rotation=270)   
            ax.set_ylabel(f'{stat.capitalize()} ({parameter})')

            ax.legend().set_visible(False)  # Hide legend on individual subplots to avoid clutter

        # Hide any remaining empty subplots
        for j in range(i + 1, rows * cols):
            fig.delaxes(axes[j])

        # Adjust layout and add a single legend
        plt.suptitle(f'Plot of yearly {stat} in {parameter.capitalize()} across grids', fontsize=16, color="red")
        plt.tight_layout()
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=2)

        # Save the plot for the current statistic
        os.makedirs(new_dir := os.path.join(fig_dir, parameter, f'07_{plot_type}'), exist_ok=True)
        plot_filename = os.path.join(new_dir, f'{stat}_barplot_{parameter}_monsoon_winter.png')
        print(f"Saving plot to: {plot_filename}") 
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()


if parameter == 'pr':
    stats_to_plot = ['%bias'] #, 'rmse']
else:
    stats_to_plot = ['abs_bias'] 
    
plot_multiple_stats_barplots(df_yearly_season, stats=stats_to_plot, rows=4, cols=5)


#%%
##### THIS CHART IS TO COMPARE YEARLY AVERAGE OF THE LENTIS DATA OF ALL MEMBERS TO ERA5 ONLY FOR THE TEMPEARTURE DATA

plot_type='lineplot_yearly_temp'
if parameter == 'pr':
    unit = "mm/season"
else:
    unit = "°C"

def plot_yearly_comparison(df, parameter, fig_dir):
    # Define color-blind-friendly colors
    colors = {
        'LENTIS': 'chocolate',  
        'ERA5': 'royalblue',    
        'Shaded Area': '#D55E00'  
    }

    # Set the suffixes based on the parameter
    if parameter == 'pr':
        lentis_col_suffix = '_sum'
        era5_col_suffix = '_sum'
    else:
        lentis_col_suffix = '_mean'
        era5_col_suffix = '_mean'

    # Get the list of columns related to LENTIS members
    lentis_columns = [col for col in df.columns if col.startswith(f'{parameter}_LENTIS_r') and col.endswith(lentis_col_suffix)]

    # Extract the relevant columns for plotting
    years = df['year']
    lentis = df[f'{parameter}_LENTIS_members{lentis_col_suffix}{lentis_col_suffix}']
    era5 = df[f'{parameter}_ERA5{era5_col_suffix}{lentis_col_suffix}']

    # Calculate the min and max across the LENTIS members columns
    lentis_min = df[lentis_columns].min(axis=1)
    lentis_max = df[lentis_columns].max(axis=1)

    # Create a line plot with a shaded area
    plt.figure(figsize=(10, 6))

    # Shaded area for LENTIS members (min to max range)
    plt.fill_between(years, lentis_min, lentis_max, color=colors['Shaded Area'], alpha=0.3, label='LENTIS Members Range')

    # Plot the average lines for LENTIS members sum and ERA5 sum
    plt.plot(years, lentis, label='LENTIS', color=colors['LENTIS'], linestyle='-', marker='o')
    plt.plot(years, era5, label='ERA5', color=colors['ERA5'], linestyle='-', marker='s')

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel(f'{parameter} ({unit})')
    plt.title(f'Yearly {parameter.capitalize()} Comparison: LENTIS Members vs ERA5')
    plt.legend()

    # Grid and layout
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    os.makedirs(new_dir := os.path.join(fig_dir, parameter, f'08_{plot_type}'), exist_ok=True)
    plot_filename = os.path.join(new_dir, f'{parameter}_yearly_avg_comparison_with_range.png')
    print(f"Saving plot to: {plot_filename}") 
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

# Check if the parameter is not 'pr'
if parameter != 'pr':
    # Run the plot_yearly_comparison function
    plot_yearly_comparison(df_yearly_overall_mean, parameter, fig_dir)
    


#%%  
##  THIS IS THE CODE TO PLOT THE SLOPE OF THE TREND LINE FOR LENTIS MEMBERS AND ERA5 ON YEARLY BASIS

# Define parameters and colors
parameter = 'pr'
plot_type = 'lineplot_yearly'
colors = {
    'LENTIS': 'darkblue',  
    'ERA5': 'darkorange',    
    'Shaded Area': '#D55E00'  
}

# Define a colormap for ensemble members
df = df_yearly  # Assuming df_yearly is already loaded

# Extract the appropriate columns based on the parameter
if parameter == 'pr':
    lentis_col_suffix = '_sum'
    era5_col_suffix = '_sum'
else:
    lentis_col_suffix = '_mean'
    era5_col_suffix = '_mean'
    
lentis_columns = [col for col in df.columns if col.startswith(f'{parameter}_LENTIS_r') and col.endswith(lentis_col_suffix)]

# Get unique lat-lon combinations and sort them by lat and lon in descending order
unique_locs = df[['lat', 'lon']].drop_duplicates().sort_values(by=['lat', 'lon'], ascending=[False, True])

# Initialize list to store slopes
slope_data = []

# Set up the figure for subplots
rows, cols = 4, 5  # Adjust these to change the grid size
fig, axes = plt.subplots(rows, cols, figsize=(25, 15))
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

# Loop over each unique lat-lon pair
for i, (lat, lon) in enumerate(unique_locs.itertuples(index=False)):
    if i >= len(axes):
        break  # Stop if there are more locations than subplots

    # Filter the DataFrame for the current lat-lon combination
    df_loc = df[(df['lat'] == lat) & (df['lon'] == lon)]

    # Extract the relevant data for the current location
    years = df_loc['year']
    pr_era5 = df_loc[f'{parameter}_ERA5{era5_col_suffix}']
    pr_lentis_members = df_loc[f'{parameter}_LENTIS_members{era5_col_suffix}']

    # Initialize list to store slopes for the current location
    lentis_slopes = []
    era5_slope = None

    # Calculate slopes for each LENTIS ensemble member
    for j, col in enumerate(lentis_columns):
        slope, _, _, _, _ = linregress(years, df_loc[col])
        lentis_slopes.append(slope)
        slope_data.append({'lat': lat, 'lon': lon, 'source': 'LENTIS', 'member': col, 'slope': slope})
        

    # Calculate and store the slope for ERA5
    slope_era5, _, _, _, _ = linregress(years, pr_era5)
    slope_lentis_members,  _, _, _, _ = linregress(years, pr_lentis_members)
    era5_slope = slope_era5
    slope_data.append({'lat': lat, 'lon': lon, 'source': 'ERA5', 'member': 'ERA5', 'slope': slope_era5})
    slope_data.append({'lat': lat, 'lon': lon, 'source': 'LENTIS', 'member': 'LENTIS_MEMBER', 'slope': slope_lentis_members})

    # Plot histogram of LENTIS slopes for the current location
    ax = axes[i]
    # Determine the bin edges based on min and max slope values
    min_slope = np.floor(min(lentis_slopes + [era5_slope]))
    max_slope = np.ceil(max(lentis_slopes + [era5_slope]))
    bins = np.arange(min_slope, max_slope + 1, 1)  
    
    ax.hist(lentis_slopes, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

    # Add vertical lines for ERA5 slope and LENTIS average slope
    ax.axvline(x=era5_slope, color='darkorange', linestyle='--', linewidth=2, label='ERA5 Slope')
    ax.axvline(x=slope_lentis_members, color='darkblue', linestyle='-', linewidth=2, label='LENTIS Average Slope')

    ax.set_title(f'Lat: {round(lat, 2)}, Lon: {round(lon, 2)}')
    ax.set_xlabel('Slope')
    ax.set_ylabel('Frequency')

    if i == 0:  # Only add the legend to the first subplot to avoid clutter
        ax.legend()

# Hide any remaining empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
os.makedirs(new_dir := os.path.join(fig_dir, parameter, f'02_{plot_type}'), exist_ok=True)
histogram_filename = os.path.join(new_dir, f'{parameter}_slopes_histograms.png')
plt.savefig(histogram_filename, dpi=300, bbox_inches='tight')
plt.show()

# Convert slope data to DataFrame and save it
slope_df = pd.DataFrame(slope_data)
slope_df_filename = os.path.join(new_dir, f'{parameter}_slopes.csv')
slope_df.to_csv(slope_df_filename, index=False)
print(f"Slopes saved to: {slope_df_filename}")

