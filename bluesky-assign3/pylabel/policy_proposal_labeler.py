import pandas as pd
import csv


def load_data(file_path):
    """
    Load the CSV data into a pandas DataFrame
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: The loaded data
    """
    try:
        df = pd.read_csv(file_path, quotechar='"')
        print(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

def main():
    """
    Main function to orchestrate the CSV processing
    """

    file_path = "bluesky_health_urls.csv"
    
    # Load the data
    df = load_data(file_path)
    print("Total number of urls is: " + str(len(df)))

if __name__ == "__main__":
    main()

    
