import openai
import numpy as np
import pandas as pd
import json


class ProductRecommendationAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def generate_personalized_recommendations(self, user_profile, purchase_history, browsing_behavior,
                                              available_products):
        """Generate personalized product recommendations using multi-step reasoning."""

        # First, analyze user preferences and patterns
        user_analysis = self._analyze_user_behavior(user_profile, purchase_history, browsing_behavior)

        # Create product summaries for context (for a real system, this would be pre-processed)
        product_context = self._format_product_context(available_products)

        # Generate recommendations with reasoning
        prompt = f"""
        Generate personalized product recommendations based on this user analysis:

        USER ANALYSIS:
        {user_analysis}

        AVAILABLE PRODUCTS:
        {product_context}

        Using the user analysis and available products, follow these steps:
        1. Identify key preferences and interests from the user's profile and behavior
        2. Find patterns in past purchases that suggest product categories of interest
        3. Consider the user's browsing behavior to identify current interests
        4. Match these preferences to the available products
        5. Rank recommendations based on relevance and likelihood of interest

        Provide your top 5 product recommendations with a detailed explanation for each,
        including why this specific product matches the user's preferences and behavior.
        Format as a numbered list with product name and reasoning for each recommendation.
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        recommendations = response.choices[0].message.content

        return {
            "user_analysis": user_analysis,
            "recommendations": recommendations
        }

    def _analyze_user_behavior(self, user_profile, purchase_history, browsing_behavior):
        """Analyze user behavior to identify preferences and patterns."""

        prompt = f"""
        Analyze this user's behavior to identify preferences, patterns, and potential interests:

        USER PROFILE:
        {json.dumps(user_profile, indent=2)}

        PURCHASE HISTORY:
        {json.dumps(purchase_history, indent=2)}

        BROWSING BEHAVIOR:
        {json.dumps(browsing_behavior, indent=2)}

        Provide a comprehensive analysis that includes:
        1. Key demographic insights and how they might influence preferences
        2. Primary product categories of interest based on purchases and browsing
        3. Price sensitivity and typical spending patterns
        4. Brand preferences or loyalty indicators
        5. Seasonal or situational shopping patterns
        6. Potential upcoming needs based on past behavior

        Focus on extracting actionable insights for product recommendations.
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content

    def _format_product_context(self, products):
        """Format product information for inclusion in prompts."""
        context = ""
        for i, product in enumerate(products[:20]):  # Limit to 20 products for context length
            context += f"Product {i + 1}: {product['name']} - ${product['price']}\n"
            context += f"Category: {product['category']}, Brand: {product['brand']}\n"
            context += f"Description: {product['description'][:100]}...\n\n"
        return context

    def generate_explanation(self, product_id, user_profile, recommendation_context):
        """Generate a personalized explanation for why a product was recommended."""

        prompt = f"""
        Generate a personalized explanation for why this product was recommended to this specific user:

        USER PROFILE:
        {json.dumps(user_profile, indent=2)}

        PRODUCT RECOMMENDATION:
        {recommendation_context}

        Create a personalized explanation that:
        1. Connects specific product features to the user's preferences or needs
        2. References relevant past purchases or browsing behavior
        3. Highlights how this product complements items they already own
        4. Explains why this is the right time for this purchase
        5. Adds a personal touch based on the user's demographics or interests

        The explanation should feel tailored to this specific user, not generic.
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        return response.choices[0].message.content


# Example usage
import data_info
recommendation_agent = ProductRecommendationAgent(data_info.open_ai_key)

# User data
user_profile = {
    "user_id": "U98765",
    "age": 34,
    "gender": "Female",
    "location": "Seattle, WA",
    "joined_date": "2023-08-15",
    "preferences": {
        "favorite_categories": ["Kitchen", "Home Decor", "Sustainable Products"],
        "size_preferences": {"clothing": "M"}
    }
}

purchase_history = [
    {"date": "2025-02-10", "product": "Organic Cotton Throw Pillows", "category": "Home Decor", "price": 45.99},
    {"date": "2025-01-22", "product": "Stainless Steel Water Bottle", "category": "Kitchen", "price": 32.50},
    {"date": "2024-12-05", "product": "Bamboo Cutting Board Set", "category": "Kitchen", "price": 65.00},
    {"date": "2024-11-18", "product": "LED String Lights", "category": "Home Decor", "price": 28.99},
    {"date": "2024-10-30", "product": "Reusable Produce Bags", "category": "Sustainable Products", "price": 15.99}
]

browsing_behavior = [
    {"date": "2025-04-18", "viewed_products": ["Ceramic Dutch Oven", "Indoor Herb Garden Kit", "Recycled Glass Vases"]},
    {"date": "2025-04-15", "viewed_products": ["Sustainable Cookware Set", "Bamboo Bathroom Accessories"]},
    {"date": "2025-04-10",
     "viewed_products": ["Indoor Herb Garden Kit", "Minimalist Wall Clock", "Plant-based Cleaning Products"]}
]

available_products = [
    {
        "id": "P12345",
        "name": "Indoor Herb Garden Kit",
        "category": "Kitchen",
        "brand": "GreenThumb",
        "price": 59.99,
        "description": "Grow fresh herbs year-round with this self-watering indoor garden kit. Includes basil, mint, and parsley seeds, plus organic soil pods."
    },
    {
        "id": "P23456",
        "name": "Recycled Glass Vase Set",
        "category": "Home Decor",
        "brand": "EcoHome",
        "price": 42.50,
        "description": "Set of 3 vases in varying sizes made from 100% recycled glass. Each piece features a unique blue-green tint with subtle bubbles."
    },
    {
        "id": "P34567",
        "name": "Bamboo Bathroom Organizer",
        "category": "Bathroom",
        "brand": "NatureLiving",
        "price": 38.99,
        "description": "Keep your bathroom tidy with this elegant organizer featuring multiple compartments, made from sustainable bamboo with water-resistant finish."
    },
    {
        "id": "P45678",
        "name": "Sustainable Cookware Set",
        "category": "Kitchen",
        "brand": "EverGreen",
        "price": 189.99,
        "description": "5-piece cookware set made with non-toxic ceramic coating and recycled aluminum. Includes 2 frying pans, saucepan, saute pan, and dutch oven."
    },
    {
        "id": "P56789",
        "name": "Minimalist Wall Clock",
        "category": "Home Decor",
        "brand": "SimpliHome",
        "price": 49.99,
        "description": "Scandinavian-inspired wall clock with wooden frame and silent movement. Perfect for creating a tranquil atmosphere in any room."
    }
]

result = recommendation_agent.generate_personalized_recommendations(
    user_profile,
    purchase_history,
    browsing_behavior,
    available_products
)

print("USER ANALYSIS:\n", result["user_analysis"])
print("\nRECOMMENDATIONS:\n", result["recommendations"])

# Generate a specific product explanation
product_explanation = recommendation_agent.generate_explanation(
    "P12345",
    user_profile,
    "Indoor Herb Garden Kit - A self-watering system to grow fresh herbs year-round in your kitchen."
)

print("\nPERSONALIZED PRODUCT EXPLANATION:\n", product_explanation)
