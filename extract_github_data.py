import requests
import pandas as pd
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.api_keys_template import GITHUB_TOKEN

def fetch_github_repos(domain, region, max_results=100, output="data/raw/git_repos.csv"):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    # GitHub Search API: search repositories by topic and region keywords
    query = f"{domain} {region} in:name,description,topics"
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=100"
    repos = []
    fetched = 0
    page = 1

    while fetched < max_results:
        paged_url = url + f"&page={page}"
        response = requests.get(paged_url, headers=headers)
        if response.status_code != 200:
            print(f"❌ GitHub API error: {response.status_code} {response.text}")
            break
        data = response.json()
        items = data.get("items", [])
        if not items:
            break
        for repo in items:
            repos.append([
                domain,
                region,
                repo.get("name", ""),
                repo.get("full_name", ""),
                repo.get("description", ""),
                repo.get("html_url", ""),
                repo.get("stargazers_count", 0),
                repo.get("language", ""),
                repo.get("created_at", ""),
                repo.get("updated_at", ""),
                ", ".join(repo.get("topics", []))
            ])
            fetched += 1
            if fetched >= max_results:
                break
        page += 1

    df = pd.DataFrame(repos, columns=[
        "Domain", "Region", "Repo Name", "Full Name", "Description", "URL",
        "Stars", "Language", "Created At", "Updated At", "Topics"
    ])
    df.to_csv(output, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df)} GitHub repos to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract trending GitHub repositories by domain and region.")
    parser.add_argument("--domain", required=True, help="Domain/topic (e.g., 'AI', 'blockchain')")
    parser.add_argument("--region", required=True, help="Region/area keyword (e.g., 'India', 'Europe')")
    parser.add_argument("--max_results", type=int, default=100, help="Number of repos to fetch")
    args = parser.parse_args()

    fetch_github_repos(args.domain, args.region, args.max_results)