"""
Health Policy Labeler for Bluesky Posts

This module implements a policy for labeling Bluesky posts that contain
potentially misleading or harmful health information.
"""

import json
import os
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from atproto import Client

import os
from dotenv import load_dotenv


class HealthPolicyLabeler:
    """
    A labeler that implements policy for health-related misinformation.
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

    def moderate_post(self, url: str, post_text=None) -> List[str]:
        """
        Moderate a Bluesky post and return a list of applicable labels

        Args:
            url: URL of the Bluesky post
            post_text: Optional text content of the post (used if URL fetch fails)

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
        return labels

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
            keyword in text.lower()
            for keyword in [
                "vaccine",
                "vaccination",
                "vax",
                "vaxx",
                "shot",
                "jab",
                "covid",
                "coronavirus",
                "pandemic",
                "virus",
                "mrna",
                "pfizer",
                "moderna",
                "novavax",
                "astrazeneca",
                "johnson",
                "cdc",
                "who",
                "fda",
                "medical",
                "health",
            ]
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

        # Process each URL
        results = []
        for _, row in df.iterrows():
            url = row["post_url"]
            post_text = row["post_text"]

            # Get the policy labels
            policy_labels = self.moderate_post(url, post_text)

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
        test_path = os.path.join(
            self.input_dir if self.input_dir else ".", "labeled_data_loose.csv"
        )
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
    """

    pass


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


if __name__ == "__main__":
    main()
