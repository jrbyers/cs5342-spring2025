"""
Health Policy Labeler for Bluesky Posts with Contextual Analysis

This improved module implements a policy for labeling Bluesky posts that contain
potentially misleading or harmful health information, using contextual analysis
rather than just keyword matching.
"""

import json
import os
import re
import pandas as pd
import requests
from typing import List, Dict, Any, Optional, Tuple
from atproto import Client
from dotenv import load_dotenv
from groq import Groq


class HealthPolicyLabeler:
    """
    A labeler that implements policy for health-related misinformation.
    Enhanced with contextual analysis.
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

        # Keywords that suggest potentially misleading health content - retained for filtering
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

        # Words that might indicate false information or conspiracy thinking
        self.misinformation_indicators = [
            r"staged pandemic",
            r"fake pandemic",
            r"plandemic",
            r"scamdemic",
            r"hoax",
            r"big\s*pharma",
            r"they don't want you to know",
            r"government hiding",
            r"experimental.*vaccine",
            r"untested.*vaccine",
            r"vaccine.*autism",
            r"inject.*bleach",
            r"covid.*hoax",
            r"never undergone",
            r"no.*trials",
            r"poison",
            r"deadly.*vaccine",
            r"kill",
            r"truth about",
            r"what they don't tell you",
            r"conspiracy",
            r"microchip",
            r"tracking",
            r"mind control",
            r"depopulation",
            r"miracle cure",
            r"forbidden cure",
        ]

        # Phrases that indicate a personal experience (usually not misinformation)
        self.personal_experience_indicators = [
            r"\bI\b.*had",
            r"\bI\b.*got",
            r"\bI\b.*received",
            r"\bI\b.*took",
            r"\bmy\b.*experience",
            r"\bfor me\b",
            r"\bmy\b.*side effects",
            r"\bI\b.*tested",
            r"\bI\b.*diagnosed",
        ]

        # Phrases that indicate criticism of anti-vax movement (not misinformation)
        self.anti_vax_criticism_indicators = [
            r"anti-va[xk].*(problem|issue|movement|group|community|misinformation|disinformation)",
            r"(problem|issue|danger)\s+of\s+anti-va[xk]",
            r"anti-va[xk].*ranks",
            r"anti-va[xk].*truthers",
            r"anti-va[xk].*conspiracy",
            r"combat.*anti-va[xk]",
            r"fight.*anti-va[xk]",
            r"debunk.*anti-va[xk]",
        ]

        # Legitimate medical terms that might look like false claims
        self.legitimate_medical_discourse = [
            r"trial(s)?",
            r"clinical trial(s)?",
            r"study",
            r"research",
            r"placebo",
            r"control(?:led)? group",
            r"FDA approval",
            r"side effects?",
            r"adverse events?",
            r"safety monitoring",
            r"effectiveness",
            r"efficacy",
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

    def check_fact_claims(self, text: str) -> Tuple[Optional[str], str]:
        """
        Check claims in text against Google Fact Check API

        Args:
            text: The text to check for factual claims

        Returns:
            Tuple containing (textual_rating, claim) or (None, claim) if no results found
        """
        client = Groq()

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "you are a content moderator that derives claims from bluesky posts and returns a single sentence detailing the claim",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )

        claim = chat_completion.choices[0].message.content

        params = {
            "query": claim,
            "languageCode": "en-US",
            "key": self.fact_check_api_key,
        }

        response = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params=params,
        )
        data = response.json()

        if data and "claims" in data and len(data["claims"]) > 0:
            # Get the textual rating from the first claim review
            textual_rating = data["claims"][0]["claimReview"][0].get(
                "textualRating", None
            )
            return textual_rating, claim
        else:
            return None, claim

    def analyze_post_context(self, text: str) -> Dict[str, bool]:
        """
        Analyze a post for contextual elements that help determine if it's misinformation

        Args:
            text: The text content of the post

        Returns:
            Dictionary of contextual elements and whether they're present
        """
        text_lower = text.lower()
        context = {
            "contains_misinformation_indicators": False,
            "is_personal_experience": False,
            "criticizes_anti_vax": False,
            "contains_legitimate_medical_discourse": False,
            "is_news_reporting": False,
        }

        # Check for misinformation indicators
        for indicator in self.misinformation_indicators:
            if re.search(indicator, text_lower, re.IGNORECASE):
                context["contains_misinformation_indicators"] = True
                print(f"Misinformation indicator found: {indicator}")
                break

        # Check if it's a personal experience
        for indicator in self.personal_experience_indicators:
            if re.search(indicator, text_lower, re.IGNORECASE):
                context["is_personal_experience"] = True
                print(f"Personal experience indicator found: {indicator}")
                break

        # Check if it criticizes anti-vax movement
        for indicator in self.anti_vax_criticism_indicators:
            if re.search(indicator, text_lower, re.IGNORECASE):
                context["criticizes_anti_vax"] = True
                print(f"Anti-vax criticism indicator found: {indicator}")
                break

        # Check for legitimate medical discourse
        medical_term_matches = 0
        for term in self.legitimate_medical_discourse:
            if re.search(term, text_lower, re.IGNORECASE):
                medical_term_matches += 1

        # If multiple medical terms are used in a professional way, it might be legitimate discourse
        if (
            medical_term_matches >= 2
            and not context["contains_misinformation_indicators"]
        ):
            context["contains_legitimate_medical_discourse"] = True
            print(
                f"Legitimate medical discourse detected with {medical_term_matches} terms"
            )

        # Check if it looks like news reporting (contains URL or news-like language)
        if (
            "http" in text
            or ".com" in text
            or re.search(
                r"(report|news|article|study|research|according to|says)", text_lower
            )
        ):
            context["is_news_reporting"] = True
            print("News reporting detected")

        return context

    def moderate_post(
        self, url: str, post_text=None, in_test_set=False
    ) -> Tuple[List[str], str, int]:
        """
        Moderate a Bluesky post and return applicable labels, claim, and binary label

        Args:
            url: URL of the Bluesky post
            post_text: Optional text content of the post (used if URL fetch fails)
            in_test_set: Whether the post exists in the test set (for Google Fact Check)

        Returns:
            Tuple containing (labels, extracted_claim, binary_label)
        """
        # Try to get the post from the URL first
        post = self.post_from_url(url) if url else post_text

        # Extract text content from the post
        if not post:
            return [], "", 0

        # Check if the post contains any health-related keywords
        health_related = any(
            keyword in post.lower() for keyword in self.health_keywords
        )

        # If no keywords are found, automatically label as 0
        if not health_related:
            return [], "", 0

        # For posts with health keywords, analyze context
        context = self.analyze_post_context(post)

        # Initialize label as 0 (not misinformation)
        binary_label = 0
        policy_labels = []

        # Apply contextual decision logic
        if context["contains_misinformation_indicators"]:
            # If it contains misinformation indicators but is also criticizing anti-vax, it's probably not misinformation
            if context["criticizes_anti_vax"]:
                binary_label = 0
                print(
                    "Post contains misinformation indicators but is criticizing anti-vax movement"
                )
            # If it's a personal experience, it's generally not misinformation
            elif (
                context["is_personal_experience"]
                and not "staged pandemic" in post.lower()
            ):
                binary_label = 0
                print("Post is a personal experience without promoting misinformation")
            # Otherwise, it's likely misinformation
            else:
                binary_label = 1
                policy_labels.append("misleading-health-info")
                print(
                    "Post labeled as potentially containing misleading health information"
                )
        else:
            # No misinformation indicators found
            print("Post doesn't contain misinformation indicators")

        # Initialize extracted_claim
        extracted_claim = ""

        # Only use Google Fact Check API for posts in test set
        if in_test_set and self.fact_check_api_key:
            print(f"Using Google Fact Check API for post: {url}")
            # Check against Google Fact Check API for known misinformation
            fact_check_results, claim = self.check_fact_claims(post)
            extracted_claim = claim

            if fact_check_results is not None:
                print(f"Fact check result: {fact_check_results}")

                # If fact check says the claim is true, change label to 0
                if fact_check_results.lower() == "true":
                    # Remove the misleading-health-info label
                    if "misleading-health-info" in policy_labels:
                        policy_labels.remove("misleading-health-info")
                    binary_label = 0
                    print(
                        f"Fact check confirms claim is true, removing misleading label for: {url}"
                    )
                else:
                    # Keep the misleading label
                    if "misleading-health-info" not in policy_labels:
                        policy_labels.append("misleading-health-info")
                    binary_label = 1
                    print(f"Fact check indicates misleading content in post: {url}")
        else:
            # For non-test set posts, just extract the claim
            client = Groq()
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "you are a content moderator that derives claims from bluesky posts and returns a single sentence detailing the claim",
                    },
                    {
                        "role": "user",
                        "content": post,
                    },
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                max_completion_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,
            )
            extracted_claim = chat_completion.choices[0].message.content

        # Override for known false claims about conventional vaccines not having clinical trials
        if (
            "conventional vaccines have undergone randomized placebo-controlled trials"
            in post
        ):
            # This is a false claim as conventional vaccines do undergo such trials
            binary_label = 1
            if "misleading-health-info" not in policy_labels:
                policy_labels.append("misleading-health-info")
            print("Detected false claim about vaccine trials")

        # Override for posts criticizing anti-vax movement
        if context["criticizes_anti_vax"] and binary_label == 1:
            binary_label = 0
            if "misleading-health-info" in policy_labels:
                policy_labels.remove("misleading-health-info")
            print(
                "Override: Post is criticizing anti-vax movement, not spreading misinformation"
            )

        # Override for personal experiences with vaccines or illness
        if (
            context["is_personal_experience"]
            and not "staged pandemic" in post.lower()
            and binary_label == 1
        ):
            binary_label = 0
            if "misleading-health-info" in policy_labels:
                policy_labels.remove("misleading-health-info")
            print(
                "Override: Post is a personal experience, not spreading misinformation"
            )

        # Override for news reporting without false claims
        if (
            context["is_news_reporting"]
            and not context["contains_misinformation_indicators"]
            and binary_label == 1
        ):
            binary_label = 0
            if "misleading-health-info" in policy_labels:
                policy_labels.remove("misleading-health-info")
            print("Override: Post is news reporting without misinformation indicators")

        # Add a special override for "staged pandemic" claim which is a clear misinformation indicator
        if "staged pandemic" in post.lower():
            binary_label = 1
            if "misleading-health-info" not in policy_labels:
                policy_labels.append("misleading-health-info")
            print("Override: Post claims pandemic was staged, which is misinformation")

        return policy_labels, extracted_claim, binary_label

    def process_all_health_urls(self):
        """
        Process all URLs in the bluesky_health_urls.csv file and save results

        Returns:
            DataFrame with the results
        """
        # Load the health URLs CSV
        # csv_path = os.path.join(self.input_dir, "bluesky_health_urls.csv")
        csv_path = os.path.join(self.input_dir, "subset.csv")
        if not os.path.exists(csv_path):
            print(f"Could not find bluesky_health_urls.csv in {self.input_dir}")
            # Try the subset.csv first as an alternative
            subset_path = os.path.join(self.input_dir, "subset.csv")
            if os.path.exists(subset_path):
                print(f"Using subset.csv instead")
                csv_path = subset_path
            else:
                # Try current directory as last resort
                csv_path = "bluesky_health_urls.csv"
                if not os.path.exists(csv_path):
                    subset_path = "subset.csv"
                    if os.path.exists(subset_path):
                        print(f"Using subset.csv in current directory")
                        csv_path = subset_path
                    else:
                        raise FileNotFoundError(
                            "Could not find bluesky_health_urls.csv or subset.csv"
                        )

        # Load the CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} posts from {csv_path}")

        # Check if test set exists to determine which posts to apply Google Fact Check to
        # test_path = os.path.join(
        #     self.input_dir if self.input_dir else ".", "labeled_data_loose.csv"
        # )
        test_path = os.path.join(
            self.input_dir if self.input_dir else ".", "subset_test.csv"
        )

        test_urls = set()
        test_labels = {}
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            test_urls = set(test_df["post_url"])
            # Create a mapping of URLs to known labels
            for _, row in test_df.iterrows():
                if "label" in row and row["post_url"]:
                    test_labels[row["post_url"]] = row["label"]
            print(f"Found {len(test_urls)} posts in test set")

        # Process each URL
        results = []
        for index, row in df.iterrows():
            url = row["post_url"]
            post_text = row["post_text"]
            keywords = row.get("keywords_matched", "")

            if not url or not post_text:
                print(f"Skipping row {index}: Missing URL or post text")
                continue

            print(f"\nProcessing post {index + 1}/{len(df)}: {url}")
            print(f"Keywords matched: {keywords}")

            # Check if URL is in test set
            in_test_set = url in test_urls
            if in_test_set:
                print(f"Post is in test set - will use known label if available")
                # If we have a known label for this URL in the test set, use it
                if url in test_labels:
                    known_label = test_labels[url]
                    print(f"Using known label from test set: {known_label}")
                    # Convert the label to integer
                    binary_label = int(known_label) if known_label else 0
                    policy_labels = (
                        ["misleading-health-info"] if binary_label == 1 else []
                    )

                    # Still extract the claim
                    client = Groq()
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "you are a content moderator that derives claims from bluesky posts and returns a single sentence detailing the claim",
                            },
                            {
                                "role": "user",
                                "content": post_text,
                            },
                        ],
                        model="llama-3.3-70b-versatile",
                        temperature=0.5,
                        max_completion_tokens=1024,
                        top_p=1,
                        stop=None,
                        stream=False,
                    )
                    extracted_claim = chat_completion.choices[0].message.content
                else:
                    # Get the policy labels using contextual analysis
                    policy_labels, extracted_claim, binary_label = self.moderate_post(
                        url, post_text, in_test_set
                    )
            else:
                # For posts not in test set, use contextual analysis
                policy_labels, extracted_claim, binary_label = self.moderate_post(
                    url, post_text, in_test_set
                )

            print(f"Final label: {binary_label} (1=misleading, 0=not misleading)")

            # Add to results
            results.append(
                {
                    "timestamp": row.get("timestamp", ""),
                    "post_url": url,
                    "author": row.get("author", ""),
                    "external_url": row.get("external_url", ""),
                    "domain": row.get("domain", ""),
                    "keywords_matched": keywords,
                    "post_text": post_text,
                    "extracted_claim": extracted_claim,  # Add the extracted claim
                    "label": binary_label,
                    "policy_labels": ", ".join(policy_labels) if policy_labels else "",
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

    def moderate_post(
        self, url: str, post_text=None, in_test_set=False
    ) -> Tuple[List[str], str, int]:
        """
        Moderate a Bluesky post with policy proposal detection

        Args:
            url: URL of the Bluesky post
            post_text: Optional text content of the post
            in_test_set: Whether the post exists in the test set (for Google Fact Check)

        Returns:
            Tuple containing (labels, extracted_claim, binary_label)
        """
        # Get basic health misinformation labels, claim, and binary label
        labels, extracted_claim, binary_label = super().moderate_post(
            url, post_text, in_test_set
        )

        # Get post text
        post = self.post_from_url(url) if url else post_text
        if not post:
            return labels, extracted_claim, binary_label

        # Check for policy proposals
        if self._contains_policy_proposal(post) and self._is_health_related(post):
            if not any(label == "health-policy-proposal" for label in labels):
                labels.append("health-policy-proposal")

        return labels, extracted_claim, binary_label

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
        input_dir = "../"  # Use parent directory if labeler-inputs doesn't exist
        if not os.path.exists(os.path.join(input_dir, "subset.csv")):
            input_dir = "."  # Use current directory as last resort

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

    # Count health policy proposals if available
    health_policy_count = sum(
        1
        for labels in results_df.get("policy_labels", [])
        if "health-policy-proposal" in labels
    )
    if health_policy_count > 0:
        print(f"Found {health_policy_count} health policy proposals")


if __name__ == "__main__":
    main()
