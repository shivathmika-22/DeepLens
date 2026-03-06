import sys
sys.path.append(r"C:\Users\dell\dl_projects\Deeplens")

def try_import(name, import_stmt):
    try:
        exec(import_stmt, globals())
        print(f"OK: {name}")
    except Exception as e:
        print(f"FAIL: {name} -> {type(e).__name__}: {e}")

try_import('YoutubeScraper', 'from scripts.youtube_scraper import YoutubeScraper')
try_import('GithubScraper', 'from scripts.github_scraper import GithubScraper')
try_import('KaggleScraper', 'from scripts.kaggle_scraper import KaggleScraper')
print('IMPORT_TEST_DONE')