---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: rocketRiding
    language: python
    name: python3
---

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/Pitwutz/rocket-riding/blob/main/notebooks/data_exploration.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="2eec9b59" -->
# Data Exploration and Understanding

<!-- #endregion -->

<!-- #region id="7e536d10" -->
# Git setup

<!-- #endregion -->

```python id="e452baf3"
# !"/Users/peterfalterbaum/Documents/Nova/thesis local/implementation/rocket riding/update_data_understanding.sh"
```

<!-- #region id="57d9dee4" -->
# base path and colab setup

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="e817fba8" outputId="d7690d32-0b48-45db-dac0-ec8c5ebedfd0"
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Check if running on Google Colab
if 'google.colab' in str(get_ipython()):
    from google.colab import drive
    drive.mount('/content/drive')
    base_path = 'drive/MyDrive/'
    logging.info("Running on Google Colab")
else:
    base_path = '/Users/peterfalterbaum/Documents/Nova/thesis local/implementation/rocket riding'
    logging.info("Running on local environment")
```

<!-- #region id="0ada9546" -->
## 1. Imports

- Overview
- Purpose of the analysis and dataset description.

<!-- #endregion -->

```python id="a2c16e88"
# IMPORTS
# import fireducks.pandas as pd
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

<!-- #region id="60660781" -->
## 2. Data Overview

<!-- #endregion -->

<!-- #region id="7b57b2c2" -->
### 2.1 Data Loading

- Code and methods used to load the dataset.

<!-- #endregion -->

```python id="3c9ec11d"
data_base_path = base_path + "/data/"
odin_path = data_base_path + "ODiN Data"
dfs = []
df_kinds = pd.read_csv(odin_path + "/tbl_kinds.csv")
dfs.append(df_kinds)
df_persons = pd.read_csv(odin_path + "/tbl_persons.csv")
dfs.append(df_persons)
df_questions = pd.read_csv(odin_path + "/tbl_questions.csv")
dfs.append(df_questions)
df_nonserial_moves = pd.read_csv(odin_path + "/tbl_nonserial_moves.csv")
dfs.append(df_nonserial_moves)
df_question_options = pd.read_csv(odin_path + "/tbl_question_options.csv")
dfs.append(df_question_options)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="6475ede8" outputId="102feae5-699a-4ef4-faad-14a57adcea6d"
df_kinds
```

```python colab={"base_uri": "https://localhost:8080/", "height": 443} id="55835c0c" outputId="d5478801-67cd-43f9-82cc-c87749cd0bfb"
df_persons
```

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="1cb25b59" outputId="00d15e23-ab3c-4dd7-d9a1-10ae485a1401"
df_questions
```

```python colab={"base_uri": "https://localhost:8080/", "height": 652} id="14625451" outputId="95ca9fb5-1a89-441e-876f-e0e0cc165bfa"
df_nonserial_moves
```

```python colab={"base_uri": "https://localhost:8080/", "height": 441} id="f081bb0f" outputId="db1c3e2d-24b8-4812-f453-df89c2b55ad1"
df_question_options
```

<!-- #region id="01a98324" -->
### 2.2 Data Preview

- Display the first few rows of the dataset to get an initial look.

- Important location features (postal codes) are given by question_id:\
  11 (ZIP address), 169 (zip dep), 170 (zip dep abr), 178 (zip arr), 179 (zip arr abr)
- Important time features are given by question_id:\
  233, 235 (, 233, 236)
- Mode of transport features are given by question_id:\
  229, 232

<!-- #endregion -->

```python id="704495ef"
# allowed_question_ids = [11, 169, 170, 178, 179, 233, 234, 235, 236, 229, 232]
# filtered_questions = df_questions[df_questions['question_id'].isin(allowed_question_ids)]
# filtered_questions
```

```python
# Define columns of interest
columns_of_interest = ['VertPC', 'AankPC', 'VertLoc']
fixed_columns = ['movement_id', 'Person_index', 'ActDuration',
                 'Travel duration', 'Timestamp Trip departure', 'Rvm', 'Timestamp Arrival']

# Create mapping from df_questions
column_mapping = {}
for col in columns_of_interest + fixed_columns:
    # Find matching question
    matching_question = df_questions[df_questions['question_name'] == col]

    if not matching_question.empty:
        # Use question text as column name
        meaning = matching_question.iloc[0]['question_text']
        column_mapping[col] = meaning.lower().replace(' ', '_')
    else:
        column_mapping[col] = col.lower().replace(' ', '_')

# Select and rename columns
columns_to_keep = fixed_columns + columns_of_interest
filtered_df_nonserial_moves = df_nonserial_moves[columns_to_keep].rename(
    columns=column_mapping)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="3d1785ad" outputId="dd181c39-2de5-4d19-e583-7100201c9505"
# filtered_df_nonserial_moves.sort_values(by='timestamp_trip_departure', inplace=True)
filtered_df_nonserial_moves
```

```python
# Count entries where departure and arrival postal codes are not the same
different_postal_codes_count = filtered_df_nonserial_moves[filtered_df_nonserial_moves[
    'postal_code_of_departure_point'] != filtered_df_nonserial_moves['arrival_point_postal_code']].shape[0]

# Print the result
print(
    f"Number of entries with different departure and arrival postal codes: {different_postal_codes_count}")
```

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Convert timestamp columns to datetime
filtered_df_nonserial_moves['timestamp_trip_departure'] = pd.to_datetime(
    filtered_df_nonserial_moves['timestamp_trip_departure'])
filtered_df_nonserial_moves['timestamp_arrival'] = pd.to_datetime(
    filtered_df_nonserial_moves['timestamp_arrival'])

# Sort the dataframe by departure timestamp
filtered_df_nonserial_moves.sort_values(
    by='timestamp_trip_departure', inplace=True)

# Calculate the count of trips by day for departures and arrivals
departure_counts = filtered_df_nonserial_moves['timestamp_trip_departure'].dt.date.value_counts(
).sort_index()
arrival_counts = filtered_df_nonserial_moves['timestamp_arrival'].dt.date.value_counts(
).sort_index()

# Plot the count of trips by day for both departures and arrivals
plt.figure(figsize=(14, 7))

# Use scatter plots to show actual data points
plt.scatter(departure_counts.index, departure_counts.values,
            label='Departure Count', alpha=0.7)
plt.scatter(arrival_counts.index, arrival_counts.values,
            label='Arrival Count', alpha=0.7)

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.title('Daily Count of Trips (Departures and Arrivals)')
plt.legend()
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Convert timestamp columns to datetime
filtered_df_nonserial_moves['timestamp_trip_departure'] = pd.to_datetime(
    filtered_df_nonserial_moves['timestamp_trip_departure'])
filtered_df_nonserial_moves['timestamp_arrival'] = pd.to_datetime(
    filtered_df_nonserial_moves['timestamp_arrival'])

# Sort the dataframe by departure timestamp
filtered_df_nonserial_moves.sort_values(
    by='timestamp_trip_departure', inplace=True)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Weekly counts
departure_counts_weekly = filtered_df_nonserial_moves['timestamp_trip_departure'].dt.isocalendar(
).week.value_counts().sort_index()
arrival_counts_weekly = filtered_df_nonserial_moves['timestamp_arrival'].dt.isocalendar(
).week.value_counts().sort_index()

# Monthly counts
departure_counts_monthly = filtered_df_nonserial_moves['timestamp_trip_departure'].dt.to_period(
    'M').value_counts().sort_index()
arrival_counts_monthly = filtered_df_nonserial_moves['timestamp_arrival'].dt.to_period(
    'M').value_counts().sort_index()

# Plot weekly counts
ax1.scatter(departure_counts_weekly.index,
            departure_counts_weekly.values, label='Departure Count', alpha=0.7)
ax1.scatter(arrival_counts_weekly.index,
            arrival_counts_weekly.values, label='Arrival Count', alpha=0.7)
ax1.set_xlabel('Week Number')
ax1.set_ylabel('Number of Trips')
ax1.set_title('Weekly Count of Trips (Departures and Arrivals)')
ax1.legend()
ax1.grid(True)

# Plot monthly counts
ax2.scatter(departure_counts_monthly.index.astype(
    str), departure_counts_monthly.values, label='Departure Count', alpha=0.7)
ax2.scatter(arrival_counts_monthly.index.astype(str),
            arrival_counts_monthly.values, label='Arrival Count', alpha=0.7)
ax2.set_xlabel('Month')
ax2.set_ylabel('Number of Trips')
ax2.set_title('Monthly Count of Trips (Departures and Arrivals)')
ax2.legend()
ax2.grid(True)
ax2.tick_params(axis='x', rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="b2a16670" outputId="fa87af79-7634-44e2-9e60-6d7e20522878"
location_columns = ['VertPC', 'AankPC', 'Person_index',
                    'movement_id', 'Timestamp Trip departure']

location_df = filtered_df_nonserial_moves[location_columns]
location_df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="3c768f9a" outputId="17e6816e-e0ae-4ff8-8c90-2e7c99b7b72b"
arrival_counts = location_df[location_df['AankPC'] != 0].groupby(
    'AankPC').size().reset_index(name='arrival_count')

# location_df.groupby('VertPC').size().reset_index(name='departure_count')

arrival_counts.to_csv(
    data_base_path + "location/arrival_counts.csv", index=False)
arrival_counts
```

```python id="3737a9f9"
location_df.to_csv(
    data_base_path + "/location/location_df.csv", index=False)
```

```python id="8f0bd626"
netherlands_gdf = gpd.read_file(
    data_base_path + "/location/working_zips.geojson")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 579} id="57528aba" outputId="fc7a28b3-ef1e-4747-d0b8-905b2627fe9a"
netherlands_gdf
```

```python id="4dd2a126"
# Aggregate counts for arrival and departure postal codes
arrival_counts = location_df.groupby(
    'AankPC').size().reset_index(name='arrival_count')
departure_counts = location_df.groupby(
    'VertPC').size().reset_index(name='departure_count')

# Rename postal code columns to match the geojson field (pc4_code)
arrival_counts.rename(columns={'AankPC': 'pc4_code'}, inplace=True)
departure_counts.rename(columns={'VertPC': 'pc4_code'}, inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="e491e99c" outputId="982393ac-dc7e-400e-82f6-a7b4888507bd"
arrival_counts
```

```python id="d6c05480"
# Ensure the postal code types match (convert to int if necessary)
netherlands_gdf['pc4_code'] = netherlands_gdf['pc4_code'].astype(int)
arrival_counts['pc4_code'] = arrival_counts['pc4_code'].astype(int)
departure_counts['pc4_code'] = departure_counts['pc4_code'].astype(int)

# Merge the counts with the Netherlands GeoDataFrame
arrival_gdf = netherlands_gdf.merge(arrival_counts, on='pc4_code', how='right')
departure_gdf = netherlands_gdf.merge(
    departure_counts, on='pc4_code', how='right')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 579} id="0e330dc0" outputId="19d11ee4-b90c-4cb3-c9c4-6ada055a5b4d"
arrival_gdf
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="c6e943fc" outputId="266c835d-d141-43ef-9c1f-d601cc690823"
# Fill missing counts with zeros
arrival_gdf['arrival_count'] = arrival_gdf['arrival_count'].fillna(0)
departure_gdf['departure_count'] = departure_gdf['departure_count'].fillna(0)

# Compute centroids of postal code areas for plotting
arrival_gdf['centroid'] = arrival_gdf.centroid
departure_gdf['centroid'] = departure_gdf.centroid

# Create the plot
fig, ax = plt.subplots(1, figsize=(12, 10))

# Plot the base map (postal code boundaries)
netherlands_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.5)

# Plot arrival locations (blue markers) - size reflects arrival count
arrival_points = arrival_gdf[arrival_gdf['arrival_count'] > 0]
arrival_points.plot(ax=ax,
                    markersize=arrival_points['arrival_count'] * 2,
                    color='blue',
                    marker='o',
                    label='Arrival')

# Plot departure locations (red markers) - size reflects departure count
departure_points = departure_gdf[departure_gdf['departure_count'] > 0]
departure_points.plot(ax=ax,
                      markersize=departure_points['departure_count'] * 2,
                      color='red',
                      marker='o',
                      label='Departure')

plt.legend()
plt.title('Overview of Arrival and Departure Locations')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 931} id="20dcb93b" outputId="35d36dae-142a-427b-bd07-ced165f765f9"
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Aggregate counts for arrival and departure postal codes
arrival_counts = location_df.groupby(
    'AankPC').size().reset_index(name='arrival_count')
departure_counts = location_df.groupby(
    'VertPC').size().reset_index(name='departure_count')

# Remove entries with pc4 code 0
arrival_counts = arrival_counts[arrival_counts['AankPC'] != 0]
departure_counts = departure_counts[departure_counts['VertPC'] != 0]

# Rename postal code columns to match the geojson field (pc4_code)
arrival_counts.rename(columns={'AankPC': 'pc4_code'}, inplace=True)
departure_counts.rename(columns={'VertPC': 'pc4_code'}, inplace=True)

# Ensure the postal code types match (convert to int if necessary)
netherlands_gdf['pc4_code'] = netherlands_gdf['pc4_code'].astype(int)
arrival_counts['pc4_code'] = arrival_counts['pc4_code'].astype(int)
departure_counts['pc4_code'] = departure_counts['pc4_code'].astype(int)

# Merge the counts with the Netherlands GeoDataFrame
arrival_gdf = netherlands_gdf.merge(arrival_counts, on='pc4_code', how='left')
departure_gdf = netherlands_gdf.merge(
    departure_counts, on='pc4_code', how='left')

# Fill missing counts with zeros
arrival_gdf['arrival_count'] = arrival_gdf['arrival_count'].fillna(0)
departure_gdf['departure_count'] = departure_gdf['departure_count'].fillna(0)

# Compute centroids of postal code areas for plotting markers
arrival_gdf['centroid'] = arrival_gdf.centroid
departure_gdf['centroid'] = departure_gdf.centroid

# Create the plot
fig, ax = plt.subplots(1, figsize=(12, 10))

# Plot the base map (postal code boundaries)
netherlands_gdf.boundary.plot(ax=ax, color='gray', linewidth=0.5)

# Plot arrival locations using a blue colormap
arrival_points = arrival_gdf[arrival_gdf['arrival_count'] > 0]
arrival_points.plot(ax=ax,
                    column='arrival_count',
                    cmap='Blues',
                    markersize=arrival_points['arrival_count'] * 2,
                    marker='o',
                    legend=True,
                    label='Arrival')

# Plot departure locations using a red colormap
departure_points = departure_gdf[departure_gdf['departure_count'] > 0]
departure_points.plot(ax=ax,
                      column='departure_count',
                      cmap='Reds',
                      markersize=departure_points['departure_count'] * 2,
                      marker='o',
                      legend=True,
                      label='Departure')

plt.title('Overview of Arrival and Departure Locations')
plt.show()
```

<!-- #region id="e63913a5" -->
### 2.3 Data Dimensions

- Number of rows and columns in the dataset.

<!-- #endregion -->

<!-- #region id="1981f637" -->
### 2.4 Data Types

- Data types of each column and any necessary type conversions.

<!-- #endregion -->

<!-- #region id="c5ea447c" -->
## 3. Data Quality and Cleaning

<!-- #endregion -->

<!-- #region id="5ea5d706" -->
### 3.1 Missing Values

- Identify and quantify missing values in the dataset.

<!-- #endregion -->

<!-- #region id="1032399e" -->
### 3.2 Unique Values

- Count of unique values in categorical columns.

<!-- #endregion -->

<!-- #region id="fe894cca" -->
### 3.3 Outliers Detection

- Identify and visualize outliers in the dataset.

<!-- #endregion -->

<!-- #region id="ecdb6da1" -->
### 3.4 Data Cleaning

- Steps taken to clean the data, such as handling missing values or outliers.

<!-- #endregion -->

<!-- #region id="104c64e9" -->
## 4. Exploratory Data Analysis (EDA)

<!-- #endregion -->

<!-- #region id="1713f60d" -->
### 4.1 Descriptive Statistics

- Summary statistics for numerical columns.

<!-- #endregion -->

<!-- #region id="cb2a5bae" -->
### 4.2 Data Distribution

- Histograms or box plots to visualize the distribution of key features.

<!-- #endregion -->

<!-- #region id="00ae9e23" -->
### 4.3 Correlation Matrix

- Heatmap or table showing correlation between numerical features.

<!-- #endregion -->

<!-- #region id="3a9d8dba" -->
### 4.4 Categorical Data Analysis

- Frequency distribution of categorical variables.

<!-- #endregion -->

<!-- #region id="fc45a034" -->
## 5. Summary and Insights

- Key insights and observations from the data exploration.

<!-- #endregion -->

```python

```

```python
from rename_files import rename_files

rename_files("/Users/peterfalterbaum/Library/CloudStorage/OneDrive-Personal/private metime/reading/Psychology/spektrum/ausgaben")
```

```python

```
