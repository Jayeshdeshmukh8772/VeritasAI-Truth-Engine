"""
VeritasAI Rate Limiting Tracker.
Tracks API call rates per model and enforces daily limits.
"""

import json
import os
from datetime import datetime
from typing import Dict


class RateTracker:
    """Tracks rate limits for each LLM provider."""

    def __init__(self, tracker_file: str = "logs/rate_tracker.json"):
        """Initialize the rate tracker."""
        self.tracker_file = tracker_file
        os.makedirs(os.path.dirname(tracker_file), exist_ok=True)
        self._ensure_file()

    def _ensure_file(self):
        """Ensure tracker file exists."""
        if not os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'w') as f:
                json.dump({}, f)

    def _load(self) -> Dict:
        """Load current rate data."""
        try:
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def increment(self, model_name: str, tokens: int = 1) -> bool:
        """Increment call count for model."""
        data = self._load()
        today = datetime.now().strftime("%Y-%m-%d")
        if model_name not in data or data[model_name].get("date") != today:
            data[model_name] = {"date": today, "count": 0, "tokens": 0}
        data[model_name]["count"] += 1
        data[model_name]["tokens"] += tokens
        with open(self.tracker_file, 'w') as f:
            json.dump(data, f)
        return True

    def get_count(self, model_name: str) -> int:
        """Get current count for a model today."""
        data = self._load()
        today = datetime.now().strftime("%Y-%m-%d")
        if model_name not in data or data[model_name].get("date") != today:
            return 0
        return data[model_name].get("count", 0)

    def check_quota(self, provider: str, daily_limit: int) -> bool:
        """Check if provider has remaining quota."""
        return self.get_count(provider) < daily_limit