import openai
import pandas as pd
import ollama
import json
from datetime import datetime, timedelta

import data_info

model = "gemma3:1b"
class FraudDetectionAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def analyze_transaction(self, transaction, user_history):
        """Analyze a transaction using multi-step reasoning to detect potential fraud."""

        # Calculate basic features
        features = self._extract_features(transaction, user_history)

        # Use the LLM for reasoning about fraud likelihood
        prompt = f"""
        Analyze this financial transaction for potential fraud:

        CURRENT TRANSACTION:
        {json.dumps(transaction, indent=2)}

        USER HISTORY SUMMARY:
        {json.dumps(features, indent=2)}

        Think step-by-step to determine if this transaction is fraudulent:
        1. Analyze location patterns and whether the current transaction location is suspicious
        2. Evaluate transaction amount in relation to user's typical spending
        3. Consider the merchant category and if it aligns with user's normal habits
        4. Assess transaction timing and frequency compared to patterns
        5. Identify specific fraud indicators present in this transaction
        6. Provide a fraud risk score (0-100) with explanation in a json format
        """

        response = ollama.generate(model=model, prompt=prompt)

        # print(response

        # Extract the risk score using regex or parsing logic
        # For simplicity, we're returning the full analysis
        return {
            "analysis": response,
            "features": features
        }

    def _extract_features(self, transaction, history):
        """Extract relevant features from transaction history."""
        # In a real system, this would involve sophisticated feature engineering

        # Calculate average transaction amount
        amounts = [tx["amount"] for tx in history]
        avg_amount = sum(amounts) / len(amounts) if amounts else 0

        # Find transaction locations
        locations = [tx["location"] for tx in history]
        common_locations = set([loc for loc in locations if locations.count(loc) > 1])

        # Calculate transaction velocity (# of transactions in last 24 hours)
        recent_count = 0
        if history:
            current_time = datetime.fromisoformat(transaction["timestamp"])
            for tx in history:
                tx_time = datetime.fromisoformat(tx["timestamp"])
                if current_time - tx_time <= timedelta(hours=24):
                    recent_count += 1

        return {
            "avg_transaction_amount": avg_amount,
            "transaction_velocity_24h": recent_count,
            "common_locations": list(common_locations),
            "usual_merchant_categories": list(set([tx["merchant_category"] for tx in history])),
            "transaction_count_30d": len(history),
            "highest_single_amount": max(amounts) if amounts else 0,
        }


# Example usage
import data_info
fraud_detector = FraudDetectionAgent(data_info.open_ai_key)

user_history = [
    {"timestamp": "2025-04-18T10:30:00", "amount": 42.15, "merchant": "Starbucks", "merchant_category": "Food",
     "location": "New York"},
    {"timestamp": "2025-04-17T18:20:00", "amount": 125.30, "merchant": "Whole Foods", "merchant_category": "Grocery",
     "location": "New York"},
    {"timestamp": "2025-04-15T12:10:00", "amount": 85.00, "merchant": "Amazon", "merchant_category": "Retail",
     "location": "Online"},
    {"timestamp": "2025-04-12T09:15:00", "amount": 35.50, "merchant": "Starbucks", "merchant_category": "Food",
     "location": "New York"},
    {"timestamp": "2025-04-10T20:20:00", "amount": 200.00, "merchant": "Nike", "merchant_category": "Retail",
     "location": "New York"},
]

suspicious_transaction = {
    "timestamp": "2025-04-19T03:45:00",
    "amount": 2499.99,
    "merchant": "Electronics Store",
    "merchant_category": "Electronics",
    "location": "Kiev"
}

result = fraud_detector.analyze_transaction(suspicious_transaction, user_history)
print(result["analysis"])
