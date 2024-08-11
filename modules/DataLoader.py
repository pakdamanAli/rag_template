import os
import pandas as pd
import logging


class DataPreprocessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_csv_files(self):
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith(".csv")]
        self.logger.info(f"Found {len(csv_files)} CSV files in {self.folder_path}.")
        return csv_files

    def _preprocess_dataframe(self, df):
        self.logger.info(f"Preprocessing DataFrame with {len(df)} rows.")
        df["question"] = df["question"].str.strip()
        df["answer"] = df["answer"].str.strip()
        return df

    def load_and_preprocess_data(self):
        self.logger.info("Loading and preprocessing data.")
        all_data = []

        for file_name in self._load_csv_files():
            file_path = os.path.join(self.folder_path, file_name)
            self.logger.info(f"Loading file {file_name}.")
            df = pd.read_csv(file_path)
            df = self._preprocess_dataframe(df)
            all_data.append(df)

        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Combined DataFrame has {len(combined_data)} rows.")
        return combined_data
