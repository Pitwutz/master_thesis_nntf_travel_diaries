"""
Utility functions for data processing and analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from pathlib import Path
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define data directory path - using absolute path from workspace root
WORKSPACE_ROOT = Path(__file__).parent.parent
DATA_DIR = WORKSPACE_ROOT / 'data' / 'raw' / 'ODiN Data'
logging.info(f"Workspace root: {WORKSPACE_ROOT}")
logging.info(f"Data directory path: {DATA_DIR}")

# Cache for question options to avoid repeated file reads
_question_options_cache = {}


def get_question_details(column_name: str, df_questions: pd.DataFrame, df_kinds: pd.DataFrame, df_question_options: pd.DataFrame) -> str:
    """
    Given a column name (question short name), return its ID, meaning, type, available answer options,
    and the datatype used for this column in df_nonserial_moves and df_persons if available.

    Args:
        column_name (str): The short name of the question (column name).
        df_questions (pd.DataFrame): DataFrame containing question information.
        df_kinds (pd.DataFrame): DataFrame containing kind information.
        df_question_options (pd.DataFrame): DataFrame containing question options.

    Returns:
        str: A detailed description of the question including its ID, meaning, type, answer options,
             and datatypes in df_nonserial_moves and df_persons if available.
    """
    try:
        # Look up the question name in the df_questions dataframe
        question_row = df_questions[df_questions['question_name']
                                    == column_name]

        if not question_row.empty:
            question_id = question_row.iloc[0]['question_id']
            question_text = question_row.iloc[0]['question_text']
            question_kind_id = question_row.iloc[0]['kind']

            # Look up the kind meaning in the df_kinds dataframe
            kind_row = df_kinds[df_kinds['kind_id'] == question_kind_id]
            kind_meaning = kind_row.iloc[0]['kind_name'] if not kind_row.empty else "Unknown"

            # Look up the answer options in the df_question_options dataframe
            options = df_question_options[df_question_options['question_id'] == question_id]
            options_list = options['option_name'].tolist()

            return f"""
Column name: {column_name}
Question ID: {question_id}
Question: {question_text}
Type: {question_kind_id} : {kind_meaning}
Options: {', '.join(map(str, options_list))}
"""
        else:
            return f"Column name: {column_name}\nNo matching question found."
    except Exception as e:
        logging.error(
            f"Error getting question details for {column_name}: {str(e)}")
        return f"Error getting details for {column_name}"


def standardize_column_name(text: str) -> str:
    """
    Standardize a column name according to our naming convention.
    This is the single source of truth for column name standardization.
    Converts to lowercase and replaces spaces with underscores.

    Args:
        text (str): Text to be standardized into a column name

    Returns:
        str: Standardized column name in lowercase with spaces replaced by underscores
    """
    return text.lower().replace(' ', '_')


def create_column_mapping(df: pd.DataFrame, df_questions: pd.DataFrame) -> Dict[str, str]:
    """
    Create a mapping dictionary for column renaming based on question text.

    Args:
        df (pd.DataFrame): DataFrame whose columns need to be mapped
        df_questions (pd.DataFrame): DataFrame containing question information

    Returns:
        dict: Dictionary mapping original column names to new column names
    """
    column_mapping = {}
    for col in df.columns:
        # Find matching question
        matching_question = df_questions[df_questions['question_name'] == col]

        if not matching_question.empty:
            # Use question text as column name
            meaning = matching_question.iloc[0]['question_text']
            column_mapping[col] = standardize_column_name(meaning)
        else:
            column_mapping[col] = standardize_column_name(col)

    return column_mapping


def format_column_name(column_name: str) -> str:
    """
    Format a column name to a consistent naming convention.

    Args:
        column_name (str): Original column name

    Returns:
        str: Formatted column name in lowercase with underscores
    """
    return column_name.lower().replace(' ', '_').replace('/', '_')


def rename_columns(df: pd.DataFrame, df_questions: pd.DataFrame) -> pd.DataFrame:
    """
    Rename DataFrame columns based on question text or standard formatting.

    Args:
        df (pd.DataFrame): DataFrame whose columns need to be renamed
        df_questions (pd.DataFrame): DataFrame containing question information

    Returns:
        pd.DataFrame: DataFrame with renamed columns
    """
    column_mapping = create_column_mapping(df, df_questions)
    return df.rename(columns=column_mapping)


def verify_travel_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifies if the travel time column matches the calculated duration from timestamps.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing trip data with timestamp and duration columns

    Returns:
    --------
    pandas.DataFrame
        DataFrame with verified travel durations
    """
    # Calculate duration from timestamps
    df['calculated_duration'] = (
        pd.to_datetime(df['timestamp_arrival']) -
        pd.to_datetime(df['timestamp_departure'])
    ).dt.total_seconds() / 60

    # Compare with reported duration
    df['duration_matches'] = abs(
        df['travel_time_in_the_netherlands_(in_minutes)'] -
        df['calculated_duration']
    ) < 1  # Allow 1 minute difference

    return df


def get_duration_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get trips where reported duration doesn't match calculated duration.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing trip data with duration columns

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing only the mismatched trips
    """
    return df[~df['duration_matches']]


def check_and_convert(value: Union[int, float]) -> int:
    """
    Check if a value can be converted to an integer and convert it.

    Args:
        value (Union[int, float]): Value to check and convert

    Returns:
        int: Converted integer value

    Raises:
        ValueError: If the value has non-zero decimal digits
    """
    if value % 1 == 0:
        return int(value)
    else:
        raise ValueError("Decimal digits are not zero for all cases.")


def check_last_digit_not_zero(value: Union[int, float]) -> bool:
    """
    Check if the last digit of a number is not zero.

    Args:
        value (Union[int, float]): Value to check

    Returns:
        bool: True if the last digit is not zero, False otherwise
    """
    str_value = str(int(value))  # Convert to string after removing decimal
    return not str_value.endswith('0')


def generate_report(dataset_name, df):
    print(f"Generating report for {dataset_name}...")
    report = ProfileReport(
        df,
        title=f"{dataset_name.capitalize()} data Report",
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": False},
            "kendall": {"calculate": False},
            "phi_k": {"calculate": False},
            "cramers": {"calculate": False},
        },
        explorative=True,
    )
    report.to_file(f"../data/reports/{dataset_name}_data_report.html")


def get_question_info(question_id: int) -> Dict[str, Union[str, int, List[str]]]:
    """
    Get detailed information about a question using its ID.

    Args:
        question_id (int): The ID of the question to look up

    Returns:
        Dict[str, Union[str, int, List[str]]]: Dictionary containing:
            - question_id: int
            - question_name: str (abbreviation)
            - question_text: str (full text)
            - question_level: str (P/V/R/W)
            - kind_id: int
            - kind_name: str
            - options: List[str] (available options if kind_id == 0)
    """
    try:
        # Read necessary CSV files
        questions_path = DATA_DIR / 'tbl_questions.csv'
        kinds_path = DATA_DIR / 'tbl_kinds.csv'
        options_path = DATA_DIR / 'tbl_question_options.csv'

        df_questions = pd.read_csv(questions_path)
        df_kinds = pd.read_csv(kinds_path)
        df_question_options = pd.read_csv(options_path)

        # Get question details
        question_row = df_questions[df_questions['question_id']
                                    == question_id].iloc[0]

        # Get kind details
        kind_row = df_kinds[df_kinds['kind_id']
                            == question_row['kind']].iloc[0]

        # Get options if kind_id is 0 (options)
        options = []
        if kind_row['kind_id'] == 0:
            options = df_question_options[
                df_question_options['question_id'] == question_id
            ]['option_name'].tolist()

        return {
            'question_id': int(question_id),
            'question_name': question_row['question_name'],
            'question_text': question_row['question_text'],
            'question_level': question_row['question_level'],
            'kind_id': int(kind_row['kind_id']),
            'kind_name': kind_row['kind_name'],
            'options': options
        }

    except Exception as e:
        logging.error(
            f"Error getting question info for ID {question_id}: {str(e)}")
        return {}


def get_standardized_column_name(identifier: Union[int, str]) -> str:
    """
    Get the standardized column name from either a question ID, column abbreviation, or question text.
    If no matching question is found, formats the identifier using the same standardization rules.

    Args:
        identifier (Union[int, str]): Either question_id (int), question_name (str), or question_text (str)

    Returns:
        str: Standardized column name using our naming convention
    """
    try:
        questions_path = DATA_DIR / 'tbl_questions.csv'
        df_questions = pd.read_csv(questions_path)

        if isinstance(identifier, int):
            # Search by question_id
            question_row = df_questions[df_questions['question_id']
                                        == identifier]
        else:
            # Search by either question_name or question_text
            question_row = df_questions[
                (df_questions['question_name'] == identifier) |
                (df_questions['question_text'] == identifier)
            ]

        if question_row.empty:
            logging.warning(f"No question found for identifier: {identifier}")
            # Format the identifier using the same standardization rules
            return standardize_column_name(str(identifier))

        # Get the question text and standardize it
        question_text = question_row.iloc[0]['question_text']
        return standardize_column_name(question_text)

    except Exception as e:
        logging.error(
            f"Error getting standardized column name for {identifier}: {str(e)}")
        # In case of error, still try to format the identifier
        return standardize_column_name(str(identifier))


def get_option_text(question_id: int, option_id: int) -> str:
    """
    Get the text description for a given option ID of a question.

    Args:
        question_id (int): The ID of the question
        option_id (int): The ID of the option to look up

    Returns:
        str: The text description of the option, or empty string if not found
    """
    try:
        options_path = DATA_DIR / 'tbl_question_options.csv'
        df_options = pd.read_csv(options_path)

        # Find the option for this question and option ID
        option_row = df_options[
            (df_options['question_id'] == question_id) &
            # option_name contains the numeric ID as string
            (df_options['option_name'] == str(option_id))
        ]

        if option_row.empty:
            logging.error(
                f"No option found for question {question_id}, option {option_id}")
            return ""

        return option_row.iloc[0]['option_text']

    except Exception as e:
        logging.error(
            f"Error getting option text for question {question_id}, option {option_id}: {str(e)}")
        return ""


def get_options_mapping(question_id: int) -> Dict[int, str]:
    """
    Get a dictionary mapping option IDs to their text descriptions for a given question.

    Args:
        question_id (int): The ID of the question

    Returns:
        Dict[int, str]: Dictionary mapping option IDs to their text descriptions
    """
    try:
        options_path = DATA_DIR / 'tbl_question_options.csv'
        df_options = pd.read_csv(options_path)

        # Get all options for this question
        question_options = df_options[df_options['question_id'] == question_id]

        if question_options.empty:
            logging.error(f"No options found for question {question_id}")
            return {}

        # Create mapping dictionary - convert option_name to int for the keys
        return {int(option_name): option_text
                for option_name, option_text
                in zip(question_options['option_name'], question_options['option_text'])}

    except Exception as e:
        logging.error(
            f"Error getting options mapping for question {question_id}: {str(e)}")
        return {}


def plot_value_distributions(
    df: pd.DataFrame,
    max_features: int = 10,
    figsize: tuple = (12, 15),
    title_prefix: str = "Distribution of"
) -> None:
    """
    Plot value distributions for categorical features in a DataFrame.
    Uses option text labels instead of IDs when available.

    Args:
        df (pd.DataFrame): DataFrame containing the features to plot
        max_features (int, optional): Maximum number of features to plot. Defaults to 10.
        figsize (tuple, optional): Figure size (width, height). Defaults to (12, 15).
        title_prefix (str, optional): Prefix for plot titles. Defaults to "Distribution of".
    """
    # Limit the number of features to plot
    features = df.columns[:max_features]

    # Create figure with vertical subplots
    fig, axes = plt.subplots(len(features), 1, figsize=(
        figsize[0], figsize[1] * len(features) / 3))

    # Handle the case when there's only one feature
    if len(features) == 1:
        axes = [axes]

    # For each feature, create a bar plot
    for i, feature in enumerate(features):
        ax = axes[i]

        # Get value counts
        value_counts = df[feature].value_counts().sort_index()
        total_count = len(df[feature])
        sum_of_counts = value_counts.sum()

        # Try to get option text labels if this is a question ID column
        try:
            # Extract question ID from column name if possible
            question_id = None

            # Check if the column name is a standardized question name
            if feature in ['destination/purpose', 'motive', 'class_division_motif', 'main_mode_of_transport_class_movement']:
                if feature == 'destination/purpose':
                    question_id = 162
                elif feature == 'motive':
                    question_id = 163
                elif feature == 'class_division_motif':
                    question_id = 164
                elif feature == 'main_mode_of_transport_class_movement':
                    question_id = 193  # KHvm

            # If we found a question ID, get the option mapping
            if question_id:
                option_mapping = get_options_mapping(question_id)

                # Create a new index with text labels
                new_index = [option_mapping.get(
                    idx, str(idx)) for idx in value_counts.index]

                # Create a new Series with text labels
                value_counts_with_labels = pd.Series(
                    value_counts.values,
                    index=new_index
                )

                # Sort by the original index to maintain order
                # Create a mapping from text to original index for sorting
                text_to_index = {text: idx for idx,
                                 text in option_mapping.items()}

                # Sort the Series by the original index values
                value_counts_with_labels = value_counts_with_labels.reindex(
                    sorted(value_counts_with_labels.index,
                           key=lambda x: text_to_index.get(x, float('inf')))
                )

                # Plot with text labels
                sns.barplot(x=value_counts_with_labels.index,
                            y=value_counts_with_labels.values, ax=ax)

                # Rotate x-axis labels for better readability
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=45, ha='right')
            else:
                # If no mapping found, plot with original index
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
        except Exception as e:
            # If any error occurs, fall back to plotting with original index
            logging.warning(
                f"Error mapping option text for {feature}: {str(e)}")
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)

        # Set title and labels with total count information
        ax.set_title(
            f'{title_prefix} {feature}\nTotal count: {total_count:,} (Sum of bars: {sum_of_counts:,})')
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    # Print missing values analysis
    missing_values = pd.DataFrame({
        'Total Missing': df[features].isna().sum(),
        'Percentage Missing': (df[features].isna().sum() / len(df) * 100).round(2)
    })

    print("\nMissing Values Analysis:")
    print(missing_values)

    # Print zero values analysis
    zero_values = pd.DataFrame({
        'Zero Values': (df[features] == 0).sum(),
        'Percentage Zeros': ((df[features] == 0).sum() / len(df) * 100).round(2),
        'Total Rows': len(df)
    })

    print("\nZero Values Analysis:")
    print(zero_values)

    # Print total counts for each feature
    print("\nTotal Counts Analysis:")
    for feature in features:
        value_counts = df[feature].value_counts()
        print(f"\n{feature}:")
        print(f"Total rows: {len(df[feature]):,}")
        print(f"Sum of all counts: {value_counts.sum():,}")
        print(f"Unique values: {len(value_counts):,}")


def get_question_options(question_name: str) -> Dict[int, Dict[str, Union[int, str]]]:
    """
    Get the options for a given question, including min/max values for numeric ranges.
    This is particularly useful for questions like travel duration classes.

    Args:
        question_name (str): The name of the question (e.g., 'travel_duration_class_in_the_netherlands')

    Returns:
        Dict[int, Dict[str, Union[int, str]]]: Dictionary mapping option IDs to their details
    """
    try:
        # Check if we already have this in the cache
        if question_name in _question_options_cache:
            return _question_options_cache[question_name]

        # Read necessary CSV files
        questions_path = DATA_DIR / 'tbl_questions.csv'
        options_path = DATA_DIR / 'tbl_question_options.csv'

        df_questions = pd.read_csv(questions_path)
        df_options = pd.read_csv(options_path)

        # Find the question ID
        question_row = df_questions[df_questions['question_name']
                                    == question_name]
        if question_row.empty:
            logging.error(f"Question '{question_name}' not found")
            return {}

        question_id = question_row.iloc[0]['question_id']

        # Get all options for this question
        question_options = df_options[df_options['question_id'] == question_id]

        if question_options.empty:
            logging.error(
                f"No options found for question '{question_name}' (ID: {question_id})")
            return {}

        # Create a dictionary to store the options
        options_dict = {}

        # Process each option
        for _, option in question_options.iterrows():
            option_id = int(option['option_name'])
            option_text = option['option_text']

            # For travel duration classes, extract min and max values
            if question_name == 'travel_duration_class_in_the_netherlands':
                # Extract min and max values from the option text
                # Format is typically like "0-15 minutes" or "15-30 minutes"
                match = re.match(r'(\d+)-(\d+)\s+minutes?', option_text)
                if match:
                    min_val = int(match.group(1))
                    max_val = int(match.group(2))
                    options_dict[option_id] = {
                        'text': option_text,
                        'min': min_val,
                        'max': max_val
                    }
                else:
                    # Handle special cases like "More than 120 minutes"
                    match = re.match(
                        r'More than (\d+)\s+minutes?', option_text)
                    if match:
                        min_val = int(match.group(1))
                        options_dict[option_id] = {
                            'text': option_text,
                            'min': min_val,
                            'max': float('inf')  # No upper limit
                        }
                    else:
                        # If we can't parse the text, just store it as is
                        options_dict[option_id] = {
                            'text': option_text
                        }
            else:
                # For other questions, just store the text
                options_dict[option_id] = {
                    'text': option_text
                }

        # Cache the result
        _question_options_cache[question_name] = options_dict

        return options_dict

    except Exception as e:
        logging.error(
            f"Error getting options for question '{question_name}': {str(e)}")
        return {}
