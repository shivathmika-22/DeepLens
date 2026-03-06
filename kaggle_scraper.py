import subprocess
import zipfile
import os
import sys
import argparse
try:
    import pandas as pd
except Exception:
    pd = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.api_keys_template import KAGGLE_PATH
from datetime import datetime
from utils.base_scraper import BaseScraper


class KaggleScraper(BaseScraper):
    """Simple Kaggle scraper wrapper to integrate with BaseScraper interface."""
    def __init__(self):
        super().__init__("kaggle")

    def fetch_data(self, query: str, max_results: int = 10, **kwargs):
        """Fetch top dataset metadata from Kaggle without downloading full archives.

        This lightweight approach lists datasets and returns basic metadata so the
        pipeline can proceed without heavy downloads.
        """
        try:
            self.logger.info(f"Searching Kaggle for datasets matching: '{query}'")
            result = subprocess.run(
                [KAGGLE_PATH, "datasets", "list", "-s", query, "--sort-by", "hottest", "-p", str(max_results)],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.splitlines()
            dataset_lines = [ln for ln in lines if "/" in ln]

            data = []
            for line in dataset_lines[:max_results]:
                parts = line.split()
                name = parts[0]
                # Construct a minimal record
                rec = {
                    'platform': 'kaggle',
                    'type': 'dataset',
                    'title': name,
                    'description': 'Kaggle dataset',
                    'content': '',
                    'url': f'https://www.kaggle.com/{name}',
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': name
                }
                data.append(rec)

            self.logger.info(f"Found {len(data)} Kaggle datasets for query '{query}'")
            return data

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Kaggle command failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching Kaggle data: {e}")
            return []

def fetch_kaggle_data(domain, max_files=10, output="data/raw/kaggle_results.csv"):
    output_dir = "data/raw/kaggle_results.csv"
    os.makedirs(output_dir, exist_ok=True)
    print(f"🔍 Searching Kaggle for datasets in domain: {domain}...")
    search_result = subprocess.run(
        [KAGGLE_PATH, "datasets", "list", "-s", domain, "--sort-by", "hottest", "-p", str(max_files)],
        capture_output=True,
        text=True
    )
    print("\n--- Top datasets found ---")
    print(search_result.stdout)

    # Parse dataset names from search result
    dataset_lines = [line for line in search_result.stdout.splitlines() if "/" in line]
    if not dataset_lines:
        print("❌ No datasets found for the domain.")
        return

    # Download and extract each dataset
    all_rows = []
    for i, dataset_line in enumerate(dataset_lines[:max_files]):
        dataset_name = dataset_line.split()[0]
        print(f"⬇ Downloading dataset: {dataset_name}...")
        subprocess.run([KAGGLE_PATH, "datasets", "download", "-d", dataset_name, "-p", output_dir], check=True)
        zip_files = [f for f in os.listdir(output_dir) if f.endswith('.zip')]
        if not zip_files:
            print("❌ No zip file found for dataset:", dataset_name)
            continue
        zip_path = os.path.join(output_dir, zip_files[0])
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"✅ Extracted {zip_path} to {output_dir}")
        os.remove(zip_path)  # Clean up zip file

        # Load CSV files and collect rows
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        if not csv_files:
            print("❌ No CSV file found in the extracted dataset.")
            continue
        for csv_file in csv_files:
            csv_path = os.path.join(output_dir, csv_file)
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
                print(f"\n📊 Preview of {csv_file}:")
                print(df.head())
                # Add domain and source info
                df["platform"] = "kaggle"
                df["domain"] = domain
                df["source"] = dataset_name
                all_rows.append(df)
            except Exception as e:
                print(f"⚠️ Could not read {csv_file}: {e}")

            # Remove CSV after reading to avoid mixing files
            os.remove(csv_path)

    # Merge all rows and save to output
    if all_rows:
        merged_df = pd.concat(all_rows, ignore_index=True)
        merged_df.to_csv(output, index=False, encoding="utf-8")
        print(f"✅ Saved Kaggle data to {output}")
    else:
        print("⚠️ No data extracted from Kaggle datasets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from Kaggle datasets by domain.")
    parser.add_argument("--domain", required=True, help="Domain/topic (e.g., 'AI', 'finance')")
    parser.add_argument("--max_files", type=int, default=100, help="Number of top datasets to extract")
    parser.add_argument("--output", default="data/raw/kaggle_results.csv", help="Path to output CSV")
    args = parser.parse_args()