import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform



def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
   
# Load the dataset from the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract unique IDs and initialize an empty distance matrix
    unique_ids = df['ID'].unique()
    n = len(unique_ids)
    
    # Initialize a distance matrix with np.inf (infinity)
    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)
    
    # Set the diagonal to 0
    np.fill_diagonal(distance_matrix.values, 0)

    # Fill the distance matrix with known distances from the DataFrame
    for index, row in df.iterrows():
        id_from = row['ID_from']
        id_to = row['ID_to']
        distance = row['Distance']
distance_matrix.loc[id_from, id_to] = distance
        distance_matrix.loc[id_to, id_from] = distance  # Ensure symmetry

    # Use the Floyd-Warshall algorithm to calculate cumulative distances
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                # Update distance matrix for the shortest path
                distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j], 
                                                  distance_matrix.loc[i, k] + distance_matrix.loc[k, j])

    return distance_matrix






def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Initialize an empty list to hold the rows of the new DataFrame
    unrolled_data = []
    
    # Iterate through the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude same id_start to id_end
                distance = distance_matrix.loc[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    
    # Create a new DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)
    
    return unrolled_df




def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    avg_distances = df.groupby('id_start')['distance'].mean()
     reference_avg_distance = avg_distances[reference_id]
     threshold = reference_avg_distance * 0.10
	ids_within_threshold = avg_distances[
        (avg_distances >= reference_avg_distance - threshold) &
        (avg_distances <= reference_avg_distance + threshold)
    ]
    
    return ids_within_threshold.reset_index()


    


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
       def calculate_toll_rate(distance_df):
    # Define toll rate coefficients
    toll_rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate in toll_rates.items():
        distance_df[vehicle_type] = distance_df['distance'] * rate

    return distance_df




def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

      # Define days and time intervals with corresponding discount factors
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    discount_factors = {
        'Weekday': {time(0, 0): 0.8, time(10, 0): 1.2, time(18, 0): 0.8},
        'Weekend': 0.7
    }

    # Prepare a list to hold the new rows
    new_rows = []

    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        
        # For each day of the week
        for day in days:
            if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                # Weekdays: apply different discount factors based on time
                for start_time in [time(0, 0), time(10, 0), time(18, 0)]:
                    end_time = time(23, 59) if start_time != time(18, 0) else time(23, 59)
                    if start_time == time(0, 0):
                        factor = discount_factors['Weekday'][time(0, 0)]
                    elif start_time == time(10, 0):
                        factor = discount_factors['Weekday'][time(10, 0)]
                    else:
			 factor = discount_factors['Weekday'][time(18, 0)]
                    
                    # Calculate rates
                    rates = {vehicle: distance * factor for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']}
                    new_rows.append([id_start, id_end, distance, day, start_time, day, end_time] + list(rates.values()))
            
            else:
                # Weekends: apply constant discount factor
                for start_time in [time(0, 0)]:
                    end_time = time(23, 59)
                    factor = discount_factors['Weekend']
                    
                    # Calculate rates
                    rates = {vehicle: distance * factor for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']}
                    new_rows.append([id_start, id_end, distance, day, start_time, day, end_time] + list(rates.values()))
    
    # Create a new DataFrame from the list of new rows
    columns = ['id_start', 'id_end', 'distance', 'start_day', 'start_time', 'end_day', 'end_time', 'moto', 'car', 'rv', 'bus', 'truck']
    result_df = pd.DataFrame(new_rows, columns=columns)
    
    return result_df

