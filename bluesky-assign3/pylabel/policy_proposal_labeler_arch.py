import pandas as pd
import csv
import random
import os

from atproto import Client
from dotenv import load_dotenv

from label import post_from_url

load_dotenv(override=True)
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")


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


def label_data(posts_df, labeled_csv="labeled_data.csv"):
    # Check if the labeled CSV already exists
    if os.path.exists(labeled_csv):
        # Load the labeled data to continue where you left off
        labeled_df = pd.read_csv(labeled_csv)
        # Find the last labeled index
        labeled_indexes = labeled_df.index.tolist()
        # Filter out the already labeled rows from the original posts_df
        remaining_posts_df = posts_df[~posts_df.index.isin(labeled_indexes)]
        print(
            f"Resuming from post #{len(labeled_df)}. {len(remaining_posts_df)} posts remaining to label."
        )
    else:
        # If no labeled CSV, randomize the posts for labeling
        remaining_posts_df = posts_df.sample(frac=1).reset_index(drop=True)
        print(
            f"Starting new labeling session. {len(remaining_posts_df)} posts to label."
        )

    # Prepare the list to store labeled data
    labeled_data = (
        labeled_df.to_dict(orient="records") if os.path.exists(labeled_csv) else []
    )

    # Loop over the remaining DataFrame rows and ask for labels
    client = Client()
    client.login(USERNAME, PW)
    for _, row in remaining_posts_df.iterrows():
        result = post_from_url(client, row["post_url"])
        post_text = result.value.text
        post_text = post_text.replace("\n", "").replace("\r", "")
        url = row["post_url"]
        external_url = row["external_url"]
        print("\n\n\n\n\n\n")
        print(f"URL: {url}")
        print(f"Text: {post_text}")
        print(f"External url: {external_url}")
        print()
        label = input("Enter 0 if the text is fine or 1 if misleading: ")

        # Validate the input to ensure it's either 0 or 1
        while label not in ["0", "1"]:
            print("Invalid input. Please enter 0 for fine or 1 for misleading.")
            label = input("Enter 0 if the text is fine or 1 if misleading: ")

        # Append the data with the label to the list
        labeled_data.append(
            {
                "timestamp": row["timestamp"],
                "post_url": row["post_url"],
                "author": row["author"],
                "external_url": row["external_url"],
                "domain": row["domain"],
                "keywords_matched": row["keywords_matched"],
                "post_text": post_text,
                "label": int(label),
            }
        )

        # Convert the labeled data into a DataFrame after each entry
        labeled_df = pd.DataFrame(labeled_data)

        # Save the labeled data incrementally to the CSV
        labeled_df.to_csv(labeled_csv, index=False)
        print(f"Labeled data updated in {labeled_csv}. Continue labeling next post.")

    print(f"Labeling completed. All data has been saved to '{labeled_csv}'.")


def main():
    """
    Main function to orchestrate the CSV processing
    """

    file_path = "bluesky_health_urls.csv"

    # Load the data
    df = load_data(file_path)
    print("Total number of urls is: " + str(len(df)))

    label_data(df)


if __name__ == "__main__":
    main()
