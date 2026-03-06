import sys, traceback
sys.path.append(r"C:\Users\dell\dl_projects\Deeplens")

modules = ['scripts.youtube_scraper', 'scripts.github_scraper', 'scripts.kaggle_scraper']
for mod in modules:
    try:
        __import__(mod)
        print(f'OK import {mod}')
    except Exception:
        print(f'FAILED import {mod}')
        traceback.print_exc()
        print('\n---\n')
