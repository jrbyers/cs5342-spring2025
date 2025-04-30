# IMPORTANT
In order to run this code the entire github repo must be cloned from the following link:

https://github.com/jrbyers/cs5342-spring2025/tree/main

There are too many cross directory dependencies for the code to run just as a collection of files.

The slides and video for this project can be found at the following link:

https://cornell.box.com/s/zohptco4b2n3oua11kmz3k02qfssaud8

# Bluesky Health Policy Labeler

A Python tool for analyzing Bluesky posts to identify and label potentially misleading health information and policy proposals. This labeler integrates with Google Fact Check API for verification of claims.

## Features

- Detects potentially misleading health information in Bluesky posts
- Identifies health-related policy proposals
- Integrates with Google Fact Check API to verify claims (for posts in test sets)
- Extracts claims from posts using LLaMa 3.3 via Groq API
- Evaluates performance against labeled test data
- Generates comprehensive CSV output with labels and extracted claims

## Requirements

- Python 3.7+
- Bluesky account credentials
- Google Fact Check API key (optional but recommended)
- Groq API key (for claim extraction)

## Ethics Note

This tool is designed for educational purposes and should not be used for commercial or malicious purposes. It should not be trusted for complete accuracy.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/bluesky-health-policy-labeler.git
   cd bluesky-health-policy-labeler
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with the following variables:
   ```
   USERNAME = "ts0211.bsky.social"
   PW = "ts0211"
   GOOGLE_FACT_CHECK_API_KEY = "AIzaSyAz07T90AxhcnD3z0Fjdlgsh9sCHfYXpgA"
   GROQ_API_KEY="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"
   ```

## Input Files

The labeler expects the following input files in a directory (default: `labeler-inputs/`):

1. **Main Input File**: Either `subset.csv` or `bluesky_health_urls.csv` with the following columns:
   - `post_url` - URL of the Bluesky post
   - `post_text` - Content of the post
   - `timestamp` (optional) - Timestamp of the post
   - `author` (optional) - Author of the post
   - `external_url` (optional) - Any external URL referenced in the post
   - `domain` (optional) - Domain of external URL
   - `keywords_matched` (optional) - Keywords that matched in the post

2. **Test Data File** (optional): `subset_test.csv` with the following columns:
   - `post_url` - URL of the Bluesky post
   - `label` - Binary label (1 = misleading, 0 = not misleading)
   - Additional columns may be present but are not used by the labeler

## Output Files

The labeler generates the following output file:

**health_policy_results.csv** with the following columns:
- `timestamp` - Timestamp from the input file
- `post_url` - URL of the Bluesky post
- `author` - Author of the post
- `external_url` - Any external URL referenced in the post
- `domain` - Domain of external URL
- `keywords_matched` - Keywords that matched in the post
- `post_text` - Content of the post
- `extracted_claim` - Claim extracted from the post using LLaMa 3.3
- `label` - Binary label (1 = misleading, 0 = not misleading)
- `policy_labels` - Comma-separated list of policy labels (e.g., "misleading-health-info, health-policy-proposal")

## Running the Labeler

Run the main script:
```
python policy_proposal_labeler.py
```

By default, the script will:
1. Look for input files in the `labeler-inputs/` directory
2. If not found, it will check the parent directory
3. If still not found, it will check the current directory
4. Process all posts in the input file
5. Generate output in `health_policy_results.csv`
6. If a test set is found, evaluate performance against it

## Labeling Logic

The labeler applies the following criteria to identify potentially misleading health information:

1. Checks if posts contain health-related keywords
2. Analyzes for suspicious phrases, skepticism indicators, conspiracy thinking, negative health phrases, and misinformation indicators
3. Applies decision logic based on combinations of these factors
4. For posts in the test set, verifies claims against Google Fact Check API
5. Identifies policy proposals using specific phrases

## Label Types

The labeler can apply the following labels:

- `misleading-health-info`: Post contains potentially misleading health information
- `health-policy-proposal`: Post contains a health-related policy proposal

## Performance Evaluation

If a test set is provided, the labeler will evaluate its performance using:
- Accuracy
- False positive count
- False negative count

## Customization

You can customize the labeler by modifying:
- Suspicious phrases
- Skepticism indicators
- Conspiracy phrases
- Negative health phrases
- Misinformation indicators
- Health keywords
- Policy proposal phrases

## Example Output

Here's an example of what the terminal output looks like when running the labeler:

```
Loaded 100 posts from subset.csv
Found 15 posts in test set

Processing post 1/100: https://bsky.app/profile/user1.bsky.social/post/12345
Keywords matched: vaccine, covid
Post is in test set - will use Google Fact Check API if available
Using Google Fact Check API for post: https://bsky.app/profile/user1.bsky.social/post/12345
Fact check result: False
Skepticism indicator found: skeptical
Suspicious phrase found: covid.*hoax
Post labeled as potentially containing misleading health information
Final label: 1 (1=misleading, 0=not misleading)

...

Results saved to health_policy_results.csv
Evaluated 15 posts from the test set:
Accuracy: 0.8667 (13/15)
False positives: 1
False negatives: 1

Processed 100 posts
Found 23 posts with potentially misleading health information
Found 12 health policy proposals
```

## Authors

Avinash Nair, John Ryan Byers, An Chi Wu

## Acknowledgements

- Google Fact Check API
- Groq API
- Bluesky Protocol