"""Implementation of automated moderator"""

import os
import csv
from typing import List, Optional
from atproto import Client
import requests
from perception import hashers
from PIL import Image
import numpy as np
import io

T_AND_S_LABEL = "t-and-s"
DOG_LABEL = "dog"
THRESH = 0.3


class AutomatedLabeler:
    """Automated labeler implementation"""

    def __init__(self, client: Client, input_dir):
        self.client = client
        self.input_dir = input_dir
        self.dog_hashes = self.load_dog_hashes()
        self.ts_words = set()
        with open(os.path.join(input_dir, "t-and-s-words.csv"), newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if row:
                    self.ts_words.add(row[0].strip().lower())

        self.ts_domains = set()
        with open(
            os.path.join(input_dir, "t-and-s-domains.csv"), newline=""
        ) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if row:
                    self.ts_domains.add(row[0].strip().lower())

        self.news_domains = {}
        with open(os.path.join(input_dir, "news-domains.csv"), newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                domain = row["Domain"].strip().lower()
                label = row["Source"].strip()
                self.news_domains[domain] = label

    def moderate_post(self, url: str) -> List[str]:
        """
        Apply moderation to the post specified by the given url
        """
        try:
            post_text = self.get_post_text(url)
            images = self.get_post_images(url)
            if not post_text:
                return []

            labels = []
            lowered_text = post_text.lower()

            if any(word in lowered_text for word in self.ts_words):
                labels.append(T_AND_S_LABEL)

            if any(domain in lowered_text for domain in self.ts_domains):
                if T_AND_S_LABEL not in labels:
                    labels.append(T_AND_S_LABEL)

            if any(self.is_dog_image(img) for img in images):
                labels.append(DOG_LABEL)

            found_sources = set()
            for domain, label in self.news_domains.items():
                if domain in lowered_text:
                    found_sources.add(label)

            labels.extend(sorted(found_sources))

            return labels

        except Exception as e:
            print(f"Error moderating post {url}: {e}")
            return []

    def get_post_text(self, url: str) -> Optional[str]:
        """
        Extract the text content from a Bluesky post given its URL.
        This uses the ATProto API via the atproto client.
        """
        try:
            parts = url.split("/profile/")[-1].split("/post/")
            if len(parts) != 2:
                return None
            author = parts[0]
            post_id = parts[1]

            post = self.client.app.bsky.feed.get_post_thread(
                {"uri": f"at://{author}/app.bsky.feed.post/{post_id}"}
            )
            return post.thread.post.record.text
        except Exception as e:
            print(f"Failed to fetch post content from {url}: {e}")
            return None

    def load_dog_hashes(self):
        dog_hashes = []
        dog_images_dir = os.path.join(self.input_dir, "dog-list-images")
        for fname in os.listdir(dog_images_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(dog_images_dir, fname)
                image = Image.open(path).convert("RGB")
                hasher = hashers.PHash()
                hash = hasher.compute(image)
                dog_hashes.append(hash)
        return dog_hashes

    def get_post_images(self, url: str) -> List[Image.Image]:
        images = []
        try:
            parts = url.split("/profile/")[-1].split("/post/")
            if len(parts) != 2:
                return []

            author = parts[0]
            post_id = parts[1]
            post = self.client.app.bsky.feed.get_post_thread(
                {"uri": f"at://{author}/app.bsky.feed.post/{post_id}"}
            )
            embeds = getattr(post.thread.post.embed, "images", [])
            for img in embeds:
                img_url = img.fullsize
                resp = requests.get(img_url)
                if resp.status_code == 200:
                    img_data = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    images.append(img_data)
        except Exception as e:
            print(f"Error fetching images from {url}: {e}")
        return images

    def is_dog_image(self, image: Image.Image) -> bool:
        hasher = hashers.PHash()
        try:
            image = image.resize((256, 256))
            img_hash = hasher.compute(image)
            for dog_hash in self.dog_hashes:
                hamming = hasher.compute_distance(img_hash, dog_hash)
                if hamming <= THRESH:
                    return True
            return False
        except Exception as e:
            print(f"Error comparing dog image: {e}")
            return False