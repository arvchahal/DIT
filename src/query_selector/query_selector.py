import pandas as pd
import random

class QuerySelector:
    """
    QuerySelector loads multiple CSV files and iterates through all queries
    in randomized order, ensuring each item is seen a specified number of times.

    Design pattern: Iterator / Generator
    """

    def __init__(self, csv_paths, cycles=1, seed=None):
        """
        Args:
            csv_paths (list[str]): List of CSV file paths.
            cycles (int): How many times to iterate through all queries.
            seed (int, optional): Random seed for reproducibility.
        """
        self.csv_paths = csv_paths
        self.cycles = cycles
        self.seed = seed
        self._load_data()
        self._reset_order()

    def _load_data(self):
        """Load and merge all CSV files into a single DataFrame."""
        self.dataframes = [pd.read_csv(path) for path in self.csv_paths]
        for i, df in enumerate(self.dataframes):
            df["source_file"] = f"file_{i+1}"
        self.all_data = pd.concat(self.dataframes, ignore_index=True)

    def _reset_order(self):
        """Generate randomized order for each cycle."""
        if self.seed is not None:
            random.seed(self.seed)
        self.full_sequence = []
        for _ in range(self.cycles):
            shuffled = self.all_data.sample(frac=1).reset_index(drop=True)
            self.full_sequence.extend(shuffled.to_dict("records"))
        self.index = 0

    def __iter__(self):
        """Enable iteration using a for-loop."""
        return self

    def __next__(self):
        """Return next query or stop iteration."""
        if self.index >= len(self.full_sequence):
            raise StopIteration
        item = self.full_sequence[self.index]
        self.index += 1
        return item

    def reset(self):
        """Reshuffle and restart iteration."""
        self._reset_order()
