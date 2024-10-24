from typing import Dict, List

import pandas as pd
import geopy.distance import geodesic
import itertools import permutations


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
     result = []
    length = len(lst)
    
    # Iterate over the list in steps of n
    for i in range(0, length, n):
        # Get the current group
        group = []
        # Gather elements for the current group
        for j in range(i, min(i + n, length)):
            group.append(lst[j])
        
        # Manually reverse the current group and add to the result
        for j in range(len(group) - 1, -1, -1):
            result.append(group[j])
    
    return result

    


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    
    # Group strings by their length
    for string in strings:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []  # Initialize the list if the key doesn't exist
        length_dict[length].append(string)  # Append the string to the correct length group
    
    # Sort the dictionary by its keys (lengths)
    sorted_length_dict = dict(sorted(length_dict.items()))
    
    return sorted_length_dict
    

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
     def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.extend(flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


        

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Generate all permutations using itertools.permutations
    all_perms = permutations(nums)
    
    # Use set to remove duplicates
    unique_perms = set(all_perms)
    
    # Convert each permutation from tuple back to list
    return [list(perm) for perm in unique_perms]

    

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Define the regex pattern for matching the dates in the specified formats
    date_pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    
    # Find all matches in the input text
    matches = re.findall(date_pattern, text)
    
    return matches

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.

    """

	# Decode the polyline string into a list of (latitude, longitude) coordinates
    coordinates = polyline.decode(polyline_str)
 # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Calculate distance for each row, using the Haversine formula
    distances = []
    
    # Add distance calculation
    for i in range(len(df)):
        if i == 0:
            distances.append(0)  # First point has no previous point to calculate distance
        else:
            lat1, lon1 = df.iloc[i - 1]
            lat2, lon2 = df.iloc[i]
            distance = haversine(lat1, lon1, lat2, lon2)
            distances.append(distance)
    
    # Add distance column to the DataFrame
    df['distance'] = distances
    
    return df

    


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
      n = len(matrix)  # Number of rows (and columns, since it's a square matrix)

    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]  # Create an empty n x n matrix
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Create a new matrix for the final result
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate the sum of the current row and column, excluding the current element
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  # Exclude the current element

    return final_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

df['timestamp'] = pd.to_datetime(df['timestamp'])
    grouped = df.groupby(['id', 'id2'])['timestamp'].agg(['min', 'max'])
    period_covered = (grouped['max'] - grouped['min']).dt.days >= 7
    return period_covered

  
