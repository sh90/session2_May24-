import openai
import json
from datetime import datetime, timedelta

import data_info


class CustomerServiceAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        self.conversation_history = []
        self.customer_context = {}

    def process_customer_query(self, customer_id, query, customer_data=None):
        """Process a customer service query using multi-step reasoning."""

        # Update customer context if provided
        if customer_data:
            self.customer_context = customer_data

        # Add the query to conversation history
        self.conversation_history.append({"role": "customer", "content": query})

        # First, understand the intent
        intent_analysis = self._analyze_intent(query)

        # Based on intent, plan the approach
        response_plan = self._plan_response(intent_analysis, query)

        # Generate the actual response
        final_response = self._generate_response(response_plan)

        # Add the response to conversation history
        self.conversation_history.append({"role": "agent", "content": final_response})

        return {
            "response": final_response,
            "intent_analysis": intent_analysis,
            "response_plan": response_plan
        }

    def _analyze_intent(self, query):
        """Analyze the customer's intent using chain-of-thought reasoning."""
        prompt = f"""
        Analyze the customer service query below to understand the customer's intent:

        CUSTOMER QUERY: "{query}"

        Think through this step-by-step:
        1. Identify the primary issue or request in the query
        2. Determine if this is a question, complaint, request for assistance, or feedback
        3. Identify any emotional tones (frustrated, confused, angry, satisfied)
        4. Extract key products, services, or account details mentioned
        5. Determine the priority level (low, medium, high)

        Provide your analysis in JSON format with these fields:
        - primary_intent
        - intent_type (question/complaint/request/feedback)
        - emotional_tone
        - mentioned_products
        - priority_level
        - key_details
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        analysis_text = response.choices[0].message.content

        # Extract JSON from the response
        try:
            # Look for JSON structure in response
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = analysis_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # Create a basic structure as fallback
                return {
                    "primary_intent": "unknown",
                    "intent_type": "unknown",
                    "emotional_tone": "neutral",
                    "mentioned_products": [],
                    "priority_level": "medium",
                    "key_details": analysis_text
                }
        except:
            # Return a fallback structure if parsing fails
            return {
                "primary_intent": "unknown",
                "intent_type": "unknown",
                "emotional_tone": "neutral",
                "mentioned_products": [],
                "priority_level": "medium",
                "key_details": analysis_text
            }

    def _plan_response(self, intent_analysis, query):
        """Plan a structured response based on the intent analysis."""

        # Create a context string with relevant customer information
        context_str = json.dumps(self.customer_context,
                                 indent=2) if self.customer_context else "No customer context available"

        # Create a history string (only include last 3 exchanges)
        history = self.conversation_history[-6:] if len(self.conversation_history) > 6 else self.conversation_history
        history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history])

        prompt = f"""
        Plan a response to this customer service interaction:

        INTENT ANALYSIS:
        {json.dumps(intent_analysis, indent=2)}

        CUSTOMER QUERY:
        "{query}"

        CUSTOMER CONTEXT:
        {context_str}

        CONVERSATION HISTORY:
        {history_str}

        Create a detailed plan for the response by:
        1. Determining what information is needed to address the query
        2. Identifying appropriate actions to resolve the issue
        3. Planning acknowledgment of customer emotions (especially if negative)
        4. Considering follow-up questions or options to present
        5. Structuring the response for clarity and empathy

        Provide your response plan in detail, outlining main points to address.
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        return response.choices[0].message.content

    def _generate_response(self, response_plan):
        """Generate the actual customer service response based on the plan."""

        prompt = f"""
        Based on this response plan, generate a compassionate, clear, and helpful customer service message:

        RESPONSE PLAN:
        {response_plan}

        Generate a natural-sounding customer service response that addresses all points in the plan
        while maintaining a supportive, professional tone. The response should be easy to understand
        and show genuine concern for the customer's needs.
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        return response.choices[0].message.content


# Example usage
import data_info
cs_agent = CustomerServiceAgent(data_info.open_ai_key)

# Customer data
customer_data = {
    "customer_id": "C1234567",
    "name": "Sarah Johnson",
    "account_type": "Premium",
    "subscription_status": "Active",
    "billing_cycle": "Monthly",
    "last_payment_date": "2025-03-15",
    "account_creation_date": "2023-07-22",
    "recent_support_issues": [
        {"date": "2025-02-10", "topic": "Login Problems", "resolved": True},
        {"date": "2025-01-05", "topic": "Billing Question", "resolved": True}
    ]
}

# Process a customer query
query = "I've been charged twice for my April subscription and I'm really frustrated because this happened last month too. Can you please fix this and make sure it doesn't happen again?"

result = cs_agent.process_customer_query("C1234567", query, customer_data)
print("INTENT ANALYSIS:\n", json.dumps(result["intent_analysis"], indent=2))
print("\nRESPONSE PLAN:\n", result["response_plan"])
print("\nFINAL RESPONSE:\n", result["response"])
