# Processed Data Overview

This directory contains processed tensor data for multiple cities in the Netherlands, with both timebin-based and weekhour-based temporal resolutions.

## Cities Overview

### Utrecht

- **Total Trips**: 5,637
- **Unique Origins**: 46
- **Unique Destinations**: 46
- **Tensor Shapes**:
  - Timebin: 46×46×35
  - Weekhour: 46×46×168
- **Maximum Trip Counts**:
  - Timebin: 22
  - Weekhour: 9
- **Postal Code Coverage**:
  - Total Postal Codes: 46
  - Used as Origins: 46
  - Used as Destinations: 46
  - Unused Codes: 0

### 's-Gravenhage (The Hague)

- **Total Trips**: 6,371
- **Unique Origins**: 62
- **Unique Destinations**: 62
- **Tensor Shapes**:
  - Timebin: 62×62×35
  - Weekhour: 62×62×168
- **Maximum Trip Counts**:
  - Timebin: 24
  - Weekhour: 12
- **Postal Code Coverage**:
  - Total Postal Codes: 62
  - Used as Origins: 62
  - Used as Destinations: 62
  - Unused Codes: 0

### Rotterdam

- **Total Trips**: 7,223
- **Unique Origins**: 76
- **Unique Destinations**: 76
- **Tensor Shapes**:
  - Timebin: 76×76×35
  - Weekhour: 76×76×168
- **Maximum Trip Counts**:
  - Timebin: 29
  - Weekhour: 8
- **Postal Code Coverage**:
  - Total Postal Codes: 76
  - Used as Origins: 76
  - Used as Destinations: 75
  - Unused Codes: 0

## Comparison Tables

### City Comparison

| Metric               | Utrecht   | 's-Gravenhage | Rotterdam |
| -------------------- | --------- | ------------- | --------- |
| Total Trips          | 5,637     | 6,371         | 7,223     |
| Unique Origins       | 46        | 62            | 76        |
| Unique Destinations  | 46        | 62            | 76        |
| Timebin Max Count    | 22        | 24            | 29        |
| Weekhour Max Count   | 9         | 12            | 8         |
| Timebin Tensor Size  | 46×46×35  | 62×62×35      | 76×76×35  |
| Weekhour Tensor Size | 46×46×168 | 62×62×168     | 76×76×168 |
| Timebin Non-Zero     | 3,418     | 4,016         | 4,335     |
| Weekhour Non-Zero    | 4,461     | 5,004         | 5,437     |
| Timebin Density      | 4.62%     | 2.98%         | 2.14%     |
| Weekhour Density     | 1.25%     | 0.77%         | 0.56%     |
| Timebin Mean Count   | 1.65      | 1.59          | 1.67      |
| Weekhour Mean Count  | 1.26      | 1.27          | 1.33      |
| Total Postal Codes   | 46        | 62            | 76        |
| Unused Codes         | 0         | 0             | 0         |

### Tensor Type Comparison

| Feature            | Timebin Tensor    | Weekhour Tensor  |
| ------------------ | ----------------- | ---------------- |
| Time Resolution    | 5 periods per day | 24 hours per day |
| Day Coverage       | Full week         | Full week        |
| Total Time Bins    | 35 (7×5)          | 168 (7×24)       |
| Typical Max Count  | Higher (22-29)    | Lower (8-12)     |
| Use Case           | Daily patterns    | Hourly patterns  |
| Storage Size       | Smaller           | Larger           |
| Temporal Detail    | Coarse            | Fine             |
| Typical Density    | 2.14% - 4.62%     | 0.56% - 1.25%    |
| Typical Mean Count | 1.59 - 1.67       | 1.26 - 1.33      |

## Tensor Types

### Timebin Tensor

- Time Resolution: 5 periods per day (Early Morning, Morning Peak, Day, Evening Peak, Night)
- Day Coverage: Full week (7 days)
- Total Time Bins: 35 (7 days × 5 periods)
- Typical Maximum Counts: 22-29 trips
- Typical Mean Counts (non-zero): 1.59-1.67 trips
- Typical Density: 2.14% - 4.62%

### Weekhour Tensor

- Time Resolution: Hourly
- Day Coverage: Full week (7 days)
- Total Time Bins: 168 (7 days × 24 hours)
- Typical Maximum Counts: 8-12 trips
- Typical Mean Counts (non-zero): 1.26-1.33 trips
- Typical Density: 0.56% - 1.25%

## Directory Structure

Each city directory contains:

```
city_name/
├── index_mappings_city_name_timebin.json
├── index_mappings_city_name_weekhour.json
└── tensors/
    ├── csv/
    │   ├── odt_processed_city_name_timebin.csv
    │   └── odt_processed_city_name_weekhour.csv
    └── odt_processed_city_name_timebin.npz
    └── odt_processed_city_name_weekhour.npz
```

## Data Format

- **CSV Files**: Contain the raw trip counts with columns: pc4_departure, pc4_arrival, timebin/weekhour, trip_count
- **NPZ Files**: Sparse tensor format containing the tensor data and metadata
- **JSON Files**: Index mappings for converting between postal codes and tensor indices

## Notes

1. **Symmetric Spaces**: All tensors use symmetric origin-destination spaces, meaning the number of unique origins equals the number of unique destinations. This ensures:

   - Complete coverage of all possible origin-destination pairs
   - Consistent tensor dimensions for analysis
   - Better compatibility with network analysis tools

2. **Postal Code Coverage**:

   - All cities have complete coverage of their postal codes in the tensors
   - Rotterdam has one postal code that appears only as an origin but not as a destination
   - No postal codes are completely unused in any city

3. **Sparsity**: The tensors are highly sparse, with most origin-destination-time combinations having zero trips. This is handled efficiently using sparse tensor storage.

4. **Time Resolution**:

   - Timebin tensors aggregate trips into 5 periods per day
   - Weekhour tensors provide hourly resolution
   - Both cover the full week (7 days)

5. **Data Consistency**: The total number of trips remains constant between timebin and weekhour representations, only the temporal aggregation differs.

## Environment & Path Management

- For programmatic access to data, use the central config.py and PROJECT_ROOT environment variable as described in the main README.
- This ensures robust, portable path management across scripts and notebooks.
