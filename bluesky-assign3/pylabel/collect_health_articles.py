"""
Script to collect posts from Bluesky that mention health or vaccine related topics
and extract URLs to be saved in a CSV file for misinformation testing.
Compatible with atproto v0.0.30. Saves URLs immediately and avoids duplicates.
"""

import os
import re
import csv
import time
import requests
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")

# Health and vaccine related keywords to filter posts by
KEYWORDS = [
    "vaccine", "vaccination", "vaxx", "vax", "antivax", "anti-vax", 
    "pfizer", "moderna", "johnson", "astrazeneca", "novavax",
    "covid", "covid-19", "coronavirus", "pandemic", 
    "immunization", "immunity", "booster", "mrna",
    "fda", "cdc", "who", "world health", 
    "adverse reaction", "side effect", "health", "medical"
]

class PostCollector:
    def __init__(self, output_file):
        self.output_file = output_file
        self.seen_urls = set()
        self.seen_posts = set()  # Track processed posts
        self.post_count = 0
        self.url_count = 0
        
        # Create/open the CSV file
        self.csv_file = open(output_file, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        self.csv_writer.writerow([
            'timestamp', 
            'post_url', 
            'author', 
            'external_url', 
            'domain',
            'keywords_matched',
            'post_text'
        ])

        # Try to load previously seen URLs and posts if file exists
        self._load_seen_data()
    
    def _load_seen_data(self):
        """Load previously seen URLs and posts from tracking files"""
        # Try to load seen URLs
        try:
            with open('seen_urls.txt', 'r') as f:
                for line in f:
                    self.seen_urls.add(line.strip())
            print(f"Loaded {len(self.seen_urls)} previously seen URLs")
        except FileNotFoundError:
            print("No previous URL history found")
        
        # Try to load seen posts
        try:
            with open('seen_posts.txt', 'r') as f:
                for line in f:
                    self.seen_posts.add(line.strip())
            print(f"Loaded {len(self.seen_posts)} previously seen posts")
        except FileNotFoundError:
            print("No previous post history found")
    
    def _save_seen_data(self):
        """Save seen URLs and posts to tracking files"""
        with open('seen_urls.txt', 'w') as f:
            for url in self.seen_urls:
                f.write(f"{url}\n")
                
        with open('seen_posts.txt', 'w') as f:
            for post_id in self.seen_posts:
                f.write(f"{post_id}\n")
    
    def __del__(self):
        """Clean up resources when object is destroyed"""
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
        
        # Save seen data
        self._save_seen_data()
    
    def extract_urls(self, text):
        """Extract URLs from text using regex pattern"""
        # URL pattern that matches common URL formats
        url_pattern = r'https?://[^\s()<>]+(?:\.[^\s()<>]+)+(?:/[^\s<>]*)?'
        return re.findall(url_pattern, text)
    
    def contains_keywords(self, text):
        """Check if the text contains any of the keywords"""
        if not text:
            return False
            
        text_lower = text.lower()
        matched_keywords = []
        
        for keyword in KEYWORDS:
            if keyword.lower() in text_lower:
                matched_keywords.append(keyword)
                
        return matched_keywords
    
    def process_post(self, post_data):
        """
        Process a single post and save to CSV if it contains relevant keywords,
        regardless of whether it contains URLs.
        """
        try:
            # Skip if post_data is None or not a dictionary
            if not post_data or not isinstance(post_data, dict):
                return

            # Extract post URI to check if we've seen it before
            post_uri = post_data.get('uri', '')
            if not post_uri:
                return
                
            # Skip if we've already processed this post
            #if post_uri in self.seen_posts:
             #   return
                
            # Mark post as seen immediately
            self.seen_posts.add(post_uri)
            
            # Save seen posts every 20 new posts
            #if len(self.seen_posts) % 20 == 0:
            self._save_seen_data()
            
            # Extract post text from the record
            record = post_data.get('record', {})
            text = ''
            
            if record and isinstance(record, dict):
                text = record.get('text', '')
            
            if not text:
                return
                
            matched_keywords = self.contains_keywords(text)
            
            # Skip if no keywords match
            if not matched_keywords:
                return
                
            self.post_count += 1
            
            # Extract author info
            author_info = post_data.get('author', {})
            author = author_info.get('handle', 'unknown') if isinstance(author_info, dict) else 'unknown'
            
            # Parse post ID from URI
            post_id = post_uri.split('/')[-1] if '/' in post_uri else 'unknown'
            post_url = f"https://bsky.app/profile/{author}/post/{post_id}"
            
            # Record timestamp
            timestamp = datetime.now().isoformat()
            
            # Extract URLs from the post text
            urls = self.extract_urls(text)
            
            # If there are no URLs, still save the post with empty URL fields
            if not urls:
                self.csv_writer.writerow([
                    timestamp, 
                    post_url, 
                    author, 
                    '', # empty external_url
                    '', # empty domain
                    ', '.join(matched_keywords),
                    text[:200]  # Include truncated post text
                ])
                
                # Flush to ensure data is written immediately
                self.csv_file.flush()
                
                print(f"Saved post without URLs: {post_url}")
            else:
                # Process each URL in the post
                for url in urls:
                    # Skip if we've already seen this URL
                    if url in self.seen_urls:
                        continue
                        
                    # Mark URL as seen immediately
                    self.seen_urls.add(url)
                    self.url_count += 1
                    
                    # Get domain for categorization
                    domain = urlparse(url).netloc
                    
                    # Write to CSV immediately
                    self.csv_writer.writerow([
                        timestamp, 
                        post_url, 
                        author, 
                        url, 
                        domain,
                        ', '.join(matched_keywords),
                        text[:200]  # Include truncated post text
                    ])
                    
                    # Flush to ensure data is written immediately
                    self.csv_file.flush()
                    
                    print(f"Saved new URL: {url}")
            
            # Save tracking data periodically
            if self.post_count % 10 == 0:
                self._save_seen_data()
                print(f"Total: {self.post_count} relevant posts ({self.url_count} with URLs)")
        
        except Exception as e:
            print(f"Error processing post: {e}")
            print(f"Post data: {post_data}")


def get_auth_token(username, password):
    """Get authentication token using direct API call instead of client login"""
    try:
        auth_url = "https://bsky.social/xrpc/com.atproto.server.createSession"
        auth_data = {
            "identifier": username,
            "password": password
        }
        
        response = requests.post(auth_url, json=auth_data)
        response.raise_for_status()
        auth_info = response.json()
        
        # Return the access token
        return auth_info.get('accessJwt')
    except Exception as e:
        print(f"Authentication error: {e}")
        return None


def fetch_timeline(auth_token, cursor=None):
    """Fetch timeline posts using direct API call"""
    try:
        timeline_url = "https://bsky.social/xrpc/app.bsky.feed.getTimeline"
        headers = {
            "Authorization": f"Bearer {auth_token}"
        }
        
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
            
        response = requests.get(timeline_url, headers=headers, params=params)
        response.raise_for_status()
        timeline_data = response.json()
        
        return timeline_data
    except Exception as e:
        print(f"Error fetching timeline: {e}")
        return None


def fetch_search_posts(auth_token, query, cursor=None):
    """Search for posts using direct API call"""
    try:
        search_url = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
        headers = {
            "Authorization": f"Bearer {auth_token}"
        }
        
        params = {"q": query, "limit": 50}
        if cursor:
            params["cursor"] = cursor
            
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_data = response.json()
        
        return search_data
    except Exception as e:
        print(f"Error searching posts: {e}")
        return None


def main():
    """Main function to run the URL collector"""
    # Output file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"bluesky_health_urls_{timestamp}.csv"
    
    print(f"Starting Bluesky health/vaccine URL collector...")
    print(f"Results will be saved to: {output_file}")
    print(f"Filtering for keywords: {', '.join(KEYWORDS)}")
    
    # Try to get auth token directly
    auth_token = get_auth_token(USERNAME, PW)
    if not auth_token:
        print("Failed to authenticate. Exiting.")
        return
    
    print("Successfully authenticated with Bluesky")
    
    # Create post collector
    collector = PostCollector(output_file)
    
    try:
        search_count = 0
        
        while True:
            search_count += 1
            print(f"\n--- Search round {search_count} ---")
            
            # Search for health-related posts
            for keyword in KEYWORDS:
                try:
                    print(f"Searching for posts with keyword: {keyword}")
                    search_data = fetch_search_posts(auth_token, keyword)
                    
                    if search_data and 'posts' in search_data:
                        print(f"Found {len(search_data['posts'])} posts for '{keyword}'")
                        for post in search_data['posts']:
                            
                            collector.process_post(post)
                    
                    # Sleep briefly to avoid rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Error while processing search results for '{keyword}': {e}")
            
            # Also check the timeline for any missed posts
            print("Checking timeline for relevant posts...")
            timeline_data = fetch_timeline(auth_token)
            if timeline_data and 'feed' in timeline_data:
                for item in timeline_data['feed']:
                    if 'post' in item:
                        collector.process_post(item['post'])
            
            # Re-authenticate every few rounds to keep token fresh
            if search_count % 5 == 0:
                print("Refreshing authentication token...")
                auth_token = get_auth_token(USERNAME, PW)
                if not auth_token:
                    print("Failed to refresh authentication. Using existing token.")
            
            # Save seen data after each round
            collector._save_seen_data()
            
            # Wait before the next round
            wait_minutes = 5
            print(f"\nWaiting for {wait_minutes} minutes before next search batch...")
            print(f"Current stats: {collector.url_count} unique URLs from {collector.post_count} relevant posts")
            print(f"Press Ctrl+C to stop collection")
            
            for i in range(wait_minutes):
                time.sleep(60)  # 1 minute
                print(f"{wait_minutes-i-1} minutes remaining until next search...")
                
    except KeyboardInterrupt:
        print("\nCollection stopped by user")
    finally:
        # Clean up and save final data
        print("Saving final data...")
        collector._save_seen_data()
        
        if hasattr(collector, 'csv_file') and not collector.csv_file.closed:
            collector.csv_file.close()
        
        print(f"\nCollection complete!")
        print(f"Processed {collector.post_count} relevant posts")
        print(f"Collected {collector.url_count} unique URLs")
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()