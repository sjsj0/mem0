"""
Seed Script: Pre-populate Qdrant with N synthetic memories per user.
=======================================================================
Run this ONCE before realistic benchmark experiments (exp_trace_realistic.py).

How it works:
  - Generates N synthetic memory texts per user from templates
  - Uses random 768-dim unit vectors (no Ollama needed — fast bulk insert)
  - Inserts directly into Qdrant bypassing mem0's add() pipeline
  - Saves to benchmarks/data/seeded_qdrant/

Why random vectors?
  Seeding 10k × 5 users = 50k real embeddings via Ollama would take ~1 hour.
  Random unit vectors are sufficient to benchmark VS search latency at scale —
  Qdrant's ANN index scan time depends on collection size, not vector content.
  The LLM still gets realistic text payloads in its prompt.

Usage:
    cd /Users/saraagarwal/mem0
    python benchmarks/seed.py
    python benchmarks/seed.py --memories-per-user 10000 --users 5
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict

import numpy as np

for _log in ("mem0", "qdrant_client", "httpx", "openai", "httpcore", "ollama"):
    logging.getLogger(_log).setLevel(logging.WARNING)

from benchmarks.common import SEED_QDRANT_PATH, SEED_COLLECTION, SEED_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary for synthetic memory generation
# ─────────────────────────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Zoe",
]
JOBS = [
    "software engineer", "data scientist", "product manager", "UX designer",
    "DevOps engineer", "ML engineer", "frontend developer", "backend developer",
    "tech lead", "engineering manager", "security engineer", "platform engineer",
    "solutions architect", "QA engineer", "analyst", "researcher", "consultant",
    "CTO", "VP of Engineering", "staff engineer",
]
COMPANIES = [
    "Google", "Meta", "Apple", "Amazon", "Netflix", "Stripe", "Airbnb", "Uber",
    "Lyft", "LinkedIn", "Salesforce", "Adobe", "Slack", "Figma", "Notion",
    "OpenAI", "Anthropic", "Databricks", "Snowflake", "Cloudflare", "Vercel",
    "GitHub", "Atlassian", "Twilio", "Datadog",
]
CITIES = [
    "San Francisco", "New York", "Austin", "Seattle", "Boston", "Chicago",
    "London", "Berlin", "Toronto", "Singapore", "Tokyo", "Sydney",
    "Amsterdam", "Paris", "Los Angeles", "Denver", "Miami", "Atlanta",
    "Portland", "Vancouver", "Dublin", "Zurich", "Stockholm", "Barcelona",
]
HOBBIES = [
    "running", "cycling", "yoga", "rock climbing", "swimming", "hiking",
    "painting", "photography", "cooking", "baking", "guitar", "piano",
    "chess", "reading", "gaming", "woodworking", "meditation", "dancing",
    "pottery", "skiing", "surfing", "kickboxing", "journaling", "drawing",
]
LANGS = [
    "Python", "TypeScript", "Go", "Rust", "Java", "C++", "Ruby", "Swift",
    "Kotlin", "Scala", "Elixir", "Haskell",
]
PROJECTS = [
    "API redesign", "mobile app rewrite", "data pipeline", "ML inference service",
    "auth system overhaul", "payment integration", "internal dashboard",
    "search engine", "recommendation system", "microservices migration",
    "CI/CD pipeline", "database migration", "AI chatbot", "analytics platform",
    "monitoring stack", "developer portal", "onboarding flow", "billing system",
]
FOODS = [
    "sushi", "pizza", "tacos", "ramen", "pasta", "curry", "burgers",
    "Mediterranean food", "Korean BBQ", "Thai cuisine", "dim sum", "pho",
]
GENRES = [
    "sci-fi", "mystery", "thriller", "fantasy", "biography", "self-help",
    "history", "philosophy", "psychology", "economics",
]
EXERCISES = [
    "marathon", "half-marathon", "triathlon", "10K", "century bike ride",
    "Ironman", "Spartan Race", "obstacle course race",
]
DURATIONS = [
    "two weeks", "a month", "six weeks", "three months", "six months",
    "a year", "eighteen months", "two years",
]
NUMS = ["two", "three", "four", "five", "six", "seven", "eight", "ten", "twelve", "fifteen"]
PET_NAMES = ["Biscuit", "Luna", "Max", "Bella", "Charlie", "Mochi", "Pepper", "Coco", "Finn", "Daisy"]

TEMPLATES = [
    # Work & career
    "Works as a {job} at {company} in {city}.",
    "Has been at {company} for {duration}.",
    "Recently got promoted to senior {job}.",
    "Started a new job as a {job} at {company} last month.",
    "Is the tech lead on the {project} project at {company}.",
    "Shipped the {project} to production after {duration} of work.",
    "Leads a team of {num} engineers at {company}.",
    "Left {company} to join a startup as {job}.",
    "Is interviewing for a {job} role at {company}.",
    "Works remotely as a {job} and lives in {city}.",
    # Programming
    "Knows {lang} and has been using it for {duration}.",
    "Switched from {lang} to {lang2} for the {project}.",
    "Open-sourced a {lang} library last quarter.",
    "Is learning {lang} on weekends.",
    # Location & living
    "Lives in {city}.",
    "Recently moved from {city} to {city2}.",
    "Planning to relocate to {city} next year.",
    "Rents an apartment near downtown {city}.",
    "Just bought a house in the suburbs of {city}.",
    # Family
    "Has {num} kids.",
    "Got married last year.",
    "Partner works as a {job} at {company}.",
    "Has a dog named {pet} and a cat.",
    "Expecting a baby in the spring.",
    "Just became a first-time parent.",
    # Health & fitness
    "Trains for a {exercise} every morning before work.",
    "Has been doing {hobby} for {duration}.",
    "Lost {num} pounds after switching to intermittent fasting.",
    "Runs {num} miles per week.",
    "Started strength training {duration} ago.",
    "Recovered from a knee injury and is back to {hobby}.",
    # Hobbies & interests
    "Loves {hobby} and does it every weekend.",
    "Is learning to play {hobby}.",
    "Reads one book a week — currently into {genre}.",
    "Got into {hobby} during the pandemic and never stopped.",
    "Favorite cuisine is {food}.",
    "Drinks coffee in the morning and tea in the afternoon.",
    # Goals & plans
    "Goal this year is to complete a {exercise}.",
    "Plans to learn {lang} before the end of the year.",
    "Saving up to travel to Japan next spring.",
    "Wants to publish a {genre} novel someday.",
    "Enrolled in an online MBA program starting in September.",
    # Challenges
    "Struggling to maintain work-life balance with the new {project} deadline.",
    "Had a rough quarter — the {project} got delayed by {duration}.",
    "Dealing with burnout after working long hours.",
    "Taking a two-week break from social media.",
    # Misc life events
    "Just got back from a two-week trip to {city}.",
    "Attended a tech conference in {city} last month.",
    "Started volunteering at a local food bank on weekends.",
    "Adopted a {pet} named {pet2} last week.",
    "Is taking {lang} lessons online three times a week.",
    "Finished their first open-source contribution.",
    "Signed up for a {genre} book club.",
    "Switched to a standing desk and loves it.",
    "Has been meditating 10 minutes every morning for {duration}.",
    "Switched careers from {job} to {job2} two years ago.",
]


def _pick(rng, lst):
    return lst[rng.integers(len(lst))]


def generate_memory_texts(n: int, user_seed: int) -> List[str]:
    """Generate n unique synthetic memory texts."""
    rng = np.random.default_rng(user_seed)
    texts = []
    seen = set()
    max_attempts = n * 10

    for _ in range(max_attempts):
        if len(texts) >= n:
            break
        template = _pick(rng, TEMPLATES)
        try:
            text = template.format(
                name=_pick(rng, FIRST_NAMES),
                job=_pick(rng, JOBS),
                job2=_pick(rng, JOBS),
                company=_pick(rng, COMPANIES),
                city=_pick(rng, CITIES),
                city2=_pick(rng, CITIES),
                hobby=_pick(rng, HOBBIES),
                lang=_pick(rng, LANGS),
                lang2=_pick(rng, LANGS),
                project=_pick(rng, PROJECTS),
                food=_pick(rng, FOODS),
                genre=_pick(rng, GENRES),
                exercise=_pick(rng, EXERCISES),
                duration=_pick(rng, DURATIONS),
                num=_pick(rng, NUMS),
                pet=_pick(rng, PET_NAMES),
                pet2=_pick(rng, PET_NAMES),
            )
        except KeyError:
            continue
        if text not in seen:
            seen.add(text)
            texts.append(text)

    return texts


def random_unit_vector(rng, dims: int = 768) -> List[float]:
    v = rng.standard_normal(dims).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────
def seed(memories_per_user: int, user_ids: List[str], batch_size: int = 500):
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct
    except ImportError:
        raise SystemExit("qdrant-client required: pip install qdrant-client")

    os.makedirs(SEED_QDRANT_PATH, exist_ok=True)
    client = QdrantClient(path=SEED_QDRANT_PATH)

    # Create collection if it doesn't exist
    existing = {c.name for c in client.get_collections().collections}
    if SEED_COLLECTION in existing:
        print(f"  Collection '{SEED_COLLECTION}' already exists — deleting and reseeding.")
        client.delete_collection(SEED_COLLECTION)

    client.create_collection(
        collection_name=SEED_COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"  Created collection '{SEED_COLLECTION}' at {SEED_QDRANT_PATH}")

    total_inserted = 0
    t_start = time.perf_counter()

    for user_id in user_ids:
        print(f"\n  Seeding user '{user_id}' with {memories_per_user:,} memories ...")
        rng = np.random.default_rng(abs(hash(user_id)) % (2**31))
        texts = generate_memory_texts(memories_per_user, abs(hash(user_id)) % (2**31))

        now = datetime.now(timezone.utc)
        points = []
        for i, text in enumerate(texts):
            mem_hash = hashlib.md5(text.encode()).hexdigest()
            created_at = (now - timedelta(days=int(rng.integers(1, 730)))).isoformat()
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=random_unit_vector(rng),
                payload={
                    "data": text,
                    "user_id": user_id,
                    "hash": mem_hash,
                    "created_at": created_at,
                    "updated_at": created_at,
                },
            ))

        # Insert in batches
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            client.upsert(collection_name=SEED_COLLECTION, points=batch)
            total_inserted += len(batch)
            print(f"    inserted {min(start + batch_size, len(points)):>6,} / {len(points):,}", end="\r")

        print(f"    inserted {len(points):,} / {len(points):,}  ✓")

    elapsed = time.perf_counter() - t_start
    total_vectors = client.get_collection(SEED_COLLECTION).points_count

    # Save seed info
    info = {
        "collection": SEED_COLLECTION,
        "path": SEED_QDRANT_PATH,
        "memories_per_user": memories_per_user,
        "users": user_ids,
        "total_vectors": total_vectors,
        "seeded_at": datetime.now(timezone.utc).isoformat(),
    }
    info_path = os.path.join(SEED_DIR, "seed_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  Total vectors in collection : {total_vectors:,}")
    print(f"  Time taken                  : {elapsed:.1f}s")
    print(f"  Seed info saved             → {info_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Seed Qdrant with synthetic memories for realistic benchmarks")
    parser.add_argument("--memories-per-user", type=int, default=10000)
    parser.add_argument("--users", type=int, default=5,
                        help="Number of users (bench_user_0 .. bench_user_N-1)")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    user_ids = [f"bench_user_{i}" for i in range(args.users)]

    print("=" * 60)
    print("Seed: Pre-populate Qdrant for Realistic Benchmarks")
    print(f"  Memories per user : {args.memories_per_user:,}")
    print(f"  Users             : {user_ids}")
    print(f"  Total vectors     : {args.memories_per_user * len(user_ids):,}")
    print(f"  Output path       : {SEED_QDRANT_PATH}")
    print("=" * 60)

    seed(
        memories_per_user=args.memories_per_user,
        user_ids=user_ids,
        batch_size=args.batch_size,
    )
    print("\n  Done. Run:  python benchmarks/exp_trace_realistic.py")


if __name__ == "__main__":
    main()
