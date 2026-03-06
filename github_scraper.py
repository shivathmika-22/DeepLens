"""
GitHub data scraper for DeepLens
"""

import os
try:
    from github import Github, GithubException, UnknownObjectException
except Exception:
    Github = None
    GithubException = Exception
    UnknownObjectException = Exception
from typing import List, Dict, Any
try:
    import pandas as pd
except Exception:
    pd = None
from datetime import datetime, timedelta
from utils.base_scraper import BaseScraper # Assuming BaseScraper is correctly imported

class GitHubScraper(BaseScraper):
    """Scraper for GitHub repositories."""
    
    def __init__(self):
        super().__init__("github")
        # --- FIX 1: Better API Key Handling ---
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            self.logger.error("GITHUB_TOKEN environment variable is not set.")
            # Raise an error or set self.github to None to handle later
            self.github = None 
        else:
            self.github = Github(github_token)
        
    def fetch_data(self, query: str, max_results: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch repositories from GitHub.
        
        Args:
            query: Search query
            max_results: Maximum number of results to fetch
            **kwargs: Additional parameters
            
        Returns:
            List of dictionaries containing repository data
        """
        if not self.github:
            self.logger.error("GitHub API client not initialized due to missing token.")
            return []
            
        self.logger.info(f"Fetching GitHub repositories for query: '{query}'")
        
        results = []
        try:
            # Search repositories
            repos = self.github.search_repositories(
                query=query,
                sort="stars",
                order="desc"
            )
            
            # --- FIX 2: Iterate and Stop Efficiently ---
            # Use enumerate to track position and stop when max_results is reached
            for i, repo in enumerate(repos):
                if i >= max_results:
                    break
                    
                # Get repo details
                repo_data = {
                    'platform': 'github',
                    'type': 'repository',
                    'title': repo.name,
                    'description': repo.description or '',
                    'content': repo.description or '', # Default content to description
                    'url': repo.html_url,
                    'timestamp': repo.created_at,
                    'stars': repo.stargazers_count,
                    'forks': repo.forks_count,
                    'language': repo.language or 'N/A', # Handle None language
                    'engagement_likes': repo.stargazers_count,
                    'engagement_shares': repo.forks_count,
                    
                    # Ensure get_topics() call doesn't raise an exception if not available
                    'topics': ', '.join(repo.get_topics()) if hasattr(repo, 'get_topics') else ''
                }
                
                # --- FIX 3: Catch specific exceptions for ReadMe ---
                try:
                    readme = repo.get_readme()
                    # The content is base64 encoded and needs to be decoded to bytes, then decoded to string
                    repo_data['content'] = readme.decoded_content.decode('utf-8')
                except UnknownObjectException:
                    # Occurs when no README file exists (404 error)
                    self.logger.debug(f"No README found for {repo.name}")
                except Exception as readme_e:
                    # Catch other potential errors during README fetching
                    self.logger.warning(f"Error fetching README for {repo.name}: {str(readme_e)}")
                    
                results.append(repo_data)
                    
            self.logger.info(f"Successfully fetched {len(results)} GitHub repositories")
            return results
            
        except GithubException as e:
            self.logger.error(f"GitHub API Error: {e.status} - {e.data}. Check your token and rate limits.")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching GitHub data: {str(e)}")
            raise


# Backwards-compatible alias: some modules import `GithubScraper` (different casing)
GithubScraper = GitHubScraper