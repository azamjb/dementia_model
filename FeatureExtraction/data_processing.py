import os
import pandas as pd
from baseline_features import extract_baseline_features
from pragmatic_features import extract_pragmatic_features
from synctactic_features import extract_syntactic_features


def process_transcripts(main_folder_path):
    data = []

    # Loop through each subfolder (e.g., 0-5, 5-10, 10-15)
    for subfolder in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder)

        # Ensure it's actually a directory
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(subfolder_path, filename)

                    with open(file_path, "r", encoding="utf-8") as file:
                        text = file.read()

                    features = extract_syntactic_features(text)

                    # Add label and metadata
                    features["filename"] = filename
                    features["label"] = "alzheimers"

                    data.append(features)

    df = pd.DataFrame(data)
    
    return df

# def process_transcripts(folder_path):
#     data = []

#     # Loop directly through all files in the single folder
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(folder_path, filename)

#             with open(file_path, "r", encoding="utf-8") as file:
#                 text = file.read()

#             features = extract_syntactic_features(text)

#             # Add label and metadata
#             features["filename"] = filename
#             features["label"] = "control"  # or "alzheimers", depending on the folder

#             data.append(features)

#     df = pd.DataFrame(data)
#     return df

# Test case
if __name__ == "__main__":

    folder_path = "../transcripts/dementia"
    df = process_transcripts(folder_path)
    print(df.head)

    df.to_csv("alzheimers_syntactic_features.csv", index=False)


# This script is being used to extract linguistic features from the datasets, and store in the form of csv files