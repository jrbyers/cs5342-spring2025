"""
Health Policy Labeler for Bluesky Posts with Google Fact Check API Integration

This module implements a policy for labeling Bluesky posts that contain
potentially misleading or harmful health information, enhanced with
Google Fact Check API verification.
"""

import json
import os
import re
import pandas as pd
import requests
from typing import List, Dict, Any, Optional, Tuple
from atproto import Client
from dotenv import load_dotenv


class HealthPolicyLabeler:
    """
    A labeler that implements policy for health-related misinformation.
    Enhanced with Google Fact Check API integration.
    """

    def __init__(self, client, input_dir=None):
        """
        Initialize the labeler with the Bluesky client and optional input directory

        Args:
            client: The Bluesky client
            input_dir: Directory containing input files
        """
        self.client = client
        self.input_dir = input_dir

        # Load Google Fact Check API key from environment variables
        load_dotenv(override=True)
        self.fact_check_api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

        if not self.fact_check_api_key:
            print(
                "Warning: Google Fact Check API key not found in environment variables"
            )

        # Keywords that suggest potentially misleading health content
        self.suspicious_phrases = [
            r"vaccine.*autism",
            r"inject.*bleach",
            r"covid.*hoax",
            r"pandemic.*fake",
            r"big\s*pharma",
            r"they don't want you to know",
            r"doctors won't tell you",
            r"government hiding",
            r"never undergone",
            r"no.*trials",
            r"experimental.*vaccine",
            r"untested.*vaccine",
            r"chemicals",
            r"toxins",
            r"poison",
            r"dangerous side effect",
            r"kill",
            r"deadly",
            r"truth about",
            r"what they don't tell you",
            r"conspiracy",
            r"microchip",
            r"tracking",
            r"5g",
            r"mind control",
            r"depopulation",
            r"alternative treatment",
            r"natural cure",
            r"miracle cure",
            r"forbidden cure",
            r"home remedy",
            r"ancient remedy",
            r"fake pandemic",
            r"plandemic",
            r"scamdemic",
        ]

        # Words that might indicate skepticism/questioning of established medical consensus
        self.skepticism_indicators = [
            "why",
            "how come",
            "explain",
            "so-called",
            "alleged",
            "supposedly",
            "scam",
            "lie",
            "lied",
            "lying",
            "corrupt",
            "agenda",
            "propaganda",
            "suspect",
            "question",
            "doubt",
            "skeptical",
            "suspicious",
            "hoax",
            "fake",
            "fraud",
            "sham",
        ]

        # Phrases that may indicate conspiracy thinking
        self.conspiracy_phrases = [
            "they don't want you to know",
            "they're hiding",
            "they won't tell you",
            "they lied",
            "cover up",
            "coverup",
            "cover-up",
            "conspiracy",
            "truth",
            "wake up",
            "sheep",
            "sheeple",
            "real truth",
            "hiding",
            "control",
            "deep state",
            "msm",
            "mainstream media lies",
            "gov't lies",
        ]

        # Negative phrases about vaccines and health measures
        self.negative_health_phrases = [
            "dangerous vaccine",
            "deadly vaccine",
            "harmful vaccine",
            "toxic vaccine",
            "poison jab",
            "killer vaccine",
            "harmful jab",
            "fake science",
            "false science",
            "bad science",
        ]

        # Indicators of harmful misinformation
        self.misinformation_indicators = [
            "globalist",
            "depopulation",
            "eugenics",
            "population control",
            "alternative facts",
            "freedom",
            "tyranny",
            "dictator",
            "tracking chip",
            "surveillance",
            "control device",
            "never been tested",
            "not properly tested",
        ]

        # Health-related keywords to search for fact checks
        self.health_keywords = [
            "vaccine",
            "vaccination",
            "vax",
            "vaxx",
            "anti-vax",
            "antivax",
            "covid",
            "covid-19",
            "coronavirus",
            "pandemic",
            "mrna",
            "pfizer",
            "moderna",
            "novavax",
            "astrazeneca",
            "johnson",
            "who",
            "cdc",
            "fda",
            "health",
            "medical",
            "immunity",
            "booster",
            "side effect",
        ]

    def post_from_url(self, url: str) -> Dict[str, Any]:
        """Extract data from a Bluesky post URL"""
        try:
            parts = url.strip("/").split("/")
            repo = parts[-3]
            rkey = parts[-1]

            # Get the post data
            response = self.client.app.bsky.feed.get_post_thread(
                {"uri": f"at://{repo}/app.bsky.feed.post/{rkey}"}
            )
            return response.thread.post.record.text
        except Exception as e:
            print(f"Warning: Could not fetch post from URL {url}: {e}")
            return None

    def check_fact_claims(self, text: str) -> Dict[str, Any]:
        """
        Check claims in text against Google Fact Check API

        Args:
            text: The text to check for factual claims

        Returns:
            Dictionary with fact check results or None if API key not available
        """
        if not self.fact_check_api_key:
            return None

        try:
            # Extract health-related phrases to check
            query_phrases = []

            # Look for health-related topics in the text
            for keyword in self.health_keywords:
                if keyword.lower() in text.lower():
                    # Find sentences containing this keyword
                    sentences = re.split(r"[.!?]+", text)
                    for sentence in sentences:
                        if keyword.lower() in sentence.lower():
                            # Clean the sentence
                            clean_sentence = sentence.strip()
                            if (
                                len(clean_sentence) > 10
                            ):  # Only if sentence is substantial
                                query_phrases.append(clean_sentence)

            if not query_phrases:
                return None

            # Send up to 3 most substantial phrases to Fact Check API
            query_phrases = sorted(query_phrases, key=len, reverse=True)[:3]

            results = []
            for query in query_phrases:
                api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
                params = {
                    "key": self.fact_check_api_key,
                    "query": query,
                    "languageCode": "en-US",
                }

                response = requests.get(api_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if "claims" in data and data["claims"]:
                        results.append({"query": query, "claims": data["claims"]})

            return {"results": results} if results else None

        except Exception as e:
            print(f"Error checking facts: {e}")
            return None

    def moderate_post(self, url: str, post_text=None, in_test_set=False) -> List[str]:
        """
        Moderate a Bluesky post and return a list of applicable labels

        Args:
            url: URL of the Bluesky post
            post_text: Optional text content of the post (used if URL fetch fails)
            in_test_set: Whether the post exists in the test set (for Google Fact Check)

        Returns:
            List of applicable labels
        """
        # Try to get the post from the URL first
        post = self.post_from_url(url) if url else post_text

        # Extract text content from the post
        if not post:
            return []

        # Apply labeling policy
        labels = self._evaluate_health_content(post)

        # Only use Google Fact Check API for posts in test set
        if in_test_set and self.fact_check_api_key:
            print(f"Using Google Fact Check API for post: {url}")
            # Check against Google Fact Check API for known misinformation
            fact_check_results = self.check_fact_claims(post)
            if fact_check_results and "results" in fact_check_results:
                debunked_claims = self._analyze_fact_check_results(fact_check_results)
                if debunked_claims and "misleading-health-info" not in labels:
                    print(f"Fact check found debunked claims in post: {url}")
                    labels.append("misleading-health-info")

        return labels

    def _analyze_fact_check_results(self, fact_check_results: Dict[str, Any]) -> bool:
        """
        Analyze fact check results to determine if any claims have been debunked

        Args:
            fact_check_results: Results from Google Fact Check API

        Returns:
            True if debunked claims found, False otherwise
        """
        if not fact_check_results or "results" not in fact_check_results:
            return False

        for result in fact_check_results["results"]:
            if "claims" in result:
                for claim in result["claims"]:
                    # Check rating - if claim rated as false, misleading, etc.
                    if "rating" in claim:
                        rating = claim["rating"].lower()
                        debunked_indicators = [
                            "false",
                            "mislead",
                            "incorrect",
                            "pants on fire",
                            "fake",
                            "wrong",
                            "misrepresent",
                            "mostly false",
                            "unsupported",
                            "exaggerated",
                            "debunked",
                        ]

                        for indicator in debunked_indicators:
                            if indicator in rating:
                                return True

        return False

    def _evaluate_health_content(self, text: str) -> List[str]:
        """
        Evaluate the health content of a post for potential misinformation

        Args:
            text: The text content of the post

        Returns:
            List of applicable labels
        """
        text = text.lower()
        labels = []

        # Check for suspicious phrases
        suspicious_phrase_found = False
        for phrase in self.suspicious_phrases:
            if re.search(phrase, text, re.IGNORECASE):
                suspicious_phrase_found = True
                break

        # Check for skepticism indicators
        skepticism_found = False
        for word in self.skepticism_indicators:
            if re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE):
                skepticism_found = True
                break

        # Check for conspiracy thinking
        conspiracy_thinking = False
        for phrase in self.conspiracy_phrases:
            if phrase in text.lower():
                conspiracy_thinking = True
                break

        # Check for negative health phrases
        negative_health_found = False
        for phrase in self.negative_health_phrases:
            if phrase in text.lower():
                negative_health_found = True
                break

        # Check for misinformation indicators
        misinformation_found = False
        for indicator in self.misinformation_indicators:
            if indicator in text.lower():
                misinformation_found = True
                break

        # Check for health-related keywords
        health_related = any(
            keyword in text.lower() for keyword in self.health_keywords
        )

        # Decision logic
        if health_related:
            # If the post contains suspicious health phrases and other indicators
            if (
                (suspicious_phrase_found and skepticism_found)
                or (conspiracy_thinking and health_related)
                or negative_health_found
                or (misinformation_found and health_related)
            ):
                labels.append("misleading-health-info")

        return labels

    def process_all_health_urls(self):
        """
        Process all URLs in the bluesky_health_urls.csv file and save results

        Returns:
            DataFrame with the results
        """
        # Load the health URLs CSV
        csv_path = os.path.join(self.input_dir, "bluesky_health_urls.csv")
        if not os.path.exists(csv_path):
            print(f"Could not find bluesky_health_urls.csv in {self.input_dir}")
            csv_path = "bluesky_health_urls.csv"  # Try the current directory
            if not os.path.exists(csv_path):
                raise FileNotFoundError("Could not find bluesky_health_urls.csv")

        # Load the CSV
        df = pd.read_csv(csv_path)

        # Check if test set exists to determine which posts to apply Google Fact Check to
        test_path = os.path.join(
            self.input_dir if self.input_dir else ".", "labeled_data_loose.csv"
        )

        test_urls = set()
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            test_urls = set(test_df["post_url"])
            print(f"Found {len(test_urls)} posts in test set")

        # Process each URL
        results = []
        for _, row in df.iterrows():
            url = row["post_url"]
            post_text = row["post_text"]

            # Check if URL is in test set
            in_test_set = url in test_urls

            # Get the policy labels - only use Google Fact Check API for test set posts
            policy_labels = self.moderate_post(url, post_text, in_test_set)

            # Convert to binary format
            label_value = 1 if "misleading-health-info" in policy_labels else 0

            # Add to results
            results.append(
                {
                    "timestamp": row.get("timestamp", ""),
                    "post_url": url,
                    "author": row.get("author", ""),
                    "external_url": row.get("external_url", ""),
                    "domain": row.get("domain", ""),
                    "keywords_matched": row.get("keywords_matched", ""),
                    "post_text": post_text,
                    "label": label_value,
                }
            )

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Save the results
        output_path = os.path.join(
            self.input_dir if self.input_dir else ".", "health_policy_results.csv"
        )
        results_df.to_csv(output_path, index=False)

        print(f"Results saved to {output_path}")

        # If labeled_data_loose.csv exists, evaluate against it
        if os.path.exists(test_path):
            self.evaluate_against_test_set(results_df, test_path)

        return results_df

    def evaluate_against_test_set(self, results_df, test_path):
        """
        Evaluate the results against a test set

        Args:
            results_df: DataFrame with results
            test_path: Path to the test CSV
        """
        # Load the test set
        test_df = pd.read_csv(test_path)

        # Create mappings
        url_to_predicted = dict(zip(results_df["post_url"], results_df["label"]))
        url_to_expected = dict(zip(test_df["post_url"], test_df["label"]))

        # Calculate metrics
        num_correct = 0
        total = 0
        false_positives = 0
        false_negatives = 0

        for url, expected in url_to_expected.items():
            if url in url_to_predicted:
                predicted = url_to_predicted[url]

                if expected == predicted:
                    num_correct += 1
                elif predicted == 1 and expected == 0:
                    false_positives += 1
                elif predicted == 0 and expected == 1:
                    false_negatives += 1

                total += 1

        # Print metrics
        if total > 0:
            accuracy = num_correct / total
            print(f"Evaluated {total} posts from the test set:")
            print(f"Accuracy: {accuracy:.4f} ({num_correct}/{total})")
            print(f"False positives: {false_positives}")
            print(f"False negatives: {false_negatives}")
        else:
            print("No matching posts found in the test set for evaluation")


class PolicyProposalLabeler(HealthPolicyLabeler):
    """
    Policy Proposal Labeler class that inherits from HealthPolicyLabeler
    with additional functionality for policy proposals.
    """

    def __init__(self, client, input_dir=None):
        """Initialize the PolicyProposalLabeler"""
        super().__init__(client, input_dir)

        # Policy proposal specific phrases
        self.policy_proposal_phrases = [
            "should be banned",
            "should be illegal",
            "should be mandatory",
            "should be required",
            "ought to be banned",
            "ought to be illegal",
            "ought to be mandatory",
            "ought to be required",
            "must be banned",
            "must be illegal",
            "must be mandatory",
            "must be required",
            "needs to be banned",
            "needs to be illegal",
            "needs to be mandatory",
            "needs to be required",
            "ban all",
            "require all",
            "mandate all",
            "make it illegal",
            "make it mandatory",
            "make it required",
            "government should",
            "governments should",
            "we should ban",
            "we should mandate",
            "we should require",
        ]

    def moderate_post(self, url: str, post_text=None, in_test_set=False) -> List[str]:
        """
        Moderate a Bluesky post with policy proposal detection

        Args:
            url: URL of the Bluesky post
            post_text: Optional text content of the post
            in_test_set: Whether the post exists in the test set (for Google Fact Check)

        Returns:
            List of applicable labels
        """
        # Get basic health misinformation labels
        labels = super().moderate_post(url, post_text, in_test_set)

        # Get post text
        post = self.post_from_url(url) if url else post_text
        if not post:
            return labels

        # Check for policy proposals
        if self._contains_policy_proposal(post) and self._is_health_related(post):
            if "health-policy-proposal" not in labels:
                labels.append("health-policy-proposal")

        return labels

    def _contains_policy_proposal(self, text: str) -> bool:
        """
        Check if text contains policy proposal language

        Args:
            text: The text to check

        Returns:
            True if a policy proposal is found, False otherwise
        """
        text = text.lower()

        for phrase in self.policy_proposal_phrases:
            if phrase in text:
                return True

        return False

    def _is_health_related(self, text: str) -> bool:
        """
        Check if text is health-related

        Args:
            text: The text to check

        Returns:
            True if health-related, False otherwise
        """
        text = text.lower()

        for keyword in self.health_keywords:
            if keyword in text:
                return True

        return False


def main():
    """
    Main function to run the policy labeler
    """

    # Load environment variables
    load_dotenv(override=True)
    USERNAME = os.getenv("USERNAME")
    PW = os.getenv("PW")

    # Initialize Bluesky client
    client = Client()
    client.login(USERNAME, PW)

    # Determine input directory
    input_dir = "labeler-inputs"
    if not os.path.exists(input_dir):
        input_dir = "../"  # Use current directory if labeler-inputs doesn't exist

    # Initialize the policy labeler
    labeler = PolicyProposalLabeler(client, input_dir)

    # Process all health URLs
    results_df = labeler.process_all_health_urls()

    print(f"Processed {len(results_df)} posts")

    # Count the number of posts labeled as misleading
    misleading_count = results_df["label"].sum()
    print(
        f"Found {misleading_count} posts with potentially misleading health information"
    )

    # Count specific policy proposals
    if "health-policy-proposal" in results_df.columns:
        policy_proposals = results_df["health-policy-proposal"].sum()
        print(f"Found {policy_proposals} health policy proposals")


if __name__ == "__main__":
    main()
