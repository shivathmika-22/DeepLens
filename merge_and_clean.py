import pandas as pd
import os
import argparse

def clean_text(text):
    import re, html
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s.,:;!?\'\"-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_date(date_str):
    from dateutil import parser
    try:
        return parser.parse(date_str).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return date_str

def load_and_clean(csv_path, platform):
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["platform"] = platform
    # For GitHub, use repo name/description; for others, use title/description
    if platform == "github":
        df["title"] = df.get("Repo Name", df.get("title", ""))
        df["description"] = df.get("Description", "")
        df["published_at"] = df.get("Created At", "")
        df["url"] = df.get("URL", "")
        df["source"] = df.get("Full Name", "")
    else:
        df["title"] = df.get("Title", df.get("title", ""))
        df["description"] = df.get("Description", df.get("description", ""))
        df["published_at"] = df.get("PublishedAt", df.get("publishedAt", ""))
        df["url"] = df.get("URL", df.get("url", ""))
        df["source"] = df.get("Source", df.get("source", ""))
    df["title"] = df["title"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)
    df["published_at"] = df["published_at"].apply(normalize_date)
    return df[["platform", "title", "description", "published_at", "url", "source"]]

def main():
    parser = argparse.ArgumentParser(description="Merge, clean, and filter YouTube, GNews, and GitHub data by domain and region.")
    parser.add_argument("--domain", type=str, required=True, help="Domain/topic to filter (e.g., 'AI', 'biotech')")
    parser.add_argument("--max_items", type=int, required=True, help="Maximum number of items to save")
    parser.add_argument("--region", type=str, default="", help="Region/location to filter (e.g., 'India', 'Europe')")
    parser.add_argument("--yt_path", type=str, default="data/raw/youtube_results.csv", help="Path to YouTube CSV")
    parser.add_argument("--gnews_path", type=str, default="data/raw/gnews_articles.csv", help="Path to GNews CSV")
    parser.add_argument("--github_path", type=str, default="data/raw/git_repos.csv", help="Path to GitHub CSV")
    parser.add_argument("--output", type=str, default="data/cleaned/merged_youtube_gnews.csv", help="Output CSV path")
    args = parser.parse_args()

    yt_df = load_and_clean(args.yt_path, "youtube")
    gn_df = load_and_clean(args.gnews_path, "gnews")
    gh_df = load_and_clean(args.github_path, "github")
    merged = pd.concat([yt_df, gn_df, gh_df], ignore_index=True)

    # Remove duplicates
    merged.drop_duplicates(subset=["title", "description"], inplace=True) 

    # Filter by domain (always) and region (if provided)
    filtered = merged[
    merged["title"].str.contains(args.domain, case=False, na=False) |
    merged["description"].str.contains(args.domain, case=False, na=False) |
    (args.region and (
        merged["title"].str.contains(args.region, case=False, na=False) |
        merged["description"].str.contains(args.region, case=False, na=False)
    ))
]
    if args.region:
        filtered = filtered[
            filtered["title"].str.contains(args.region, case=False, na=False) |
            filtered["description"].str.contains(args.region, case=False, na=False)
        ]

    # Sort by published date (trending/recent)
    filtered = filtered.sort_values(by="published_at", ascending=False)

    # Save up to max_items
    final = filtered.head(args.max_items)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final.to_csv(args.output, index=False, encoding="utf-8")
    if final.empty:
        print("⚠️ No matching data found for your domain and region. Try different keywords or check your raw data.")
    else:
        print(f"✅ Merged, cleaned, and filtered dataset saved to {args.output}")

if __name__ == "__main__":
    main()