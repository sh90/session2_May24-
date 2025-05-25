import openai
import json


class FinancialAdvisorAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def provide_investment_advice(self, user_profile, investment_goals):
        """Generate personalized investment advice using chain-of-thought reasoning."""

        prompt = f"""
        As a financial advisor, provide investment recommendations for this client basis on only the information from year 2020-2024:

        CLIENT PROFILE:
        {json.dumps(user_profile, indent=2)}

        INVESTMENT GOALS:
        {investment_goals}

        Let's think through this step-by-step:
        1. Analyze the client's risk tolerance based on age, financial situation, and goals
        2. Consider current market conditions and economic factors
        3. Evaluate appropriate asset allocation (stocks, bonds, alternatives)
        4. Recommend specific investment vehicles and explain the rationale
        5. Address potential concerns and provide risk mitigation strategies
        
        """
        # response = client.responses.create(
        #     model="gpt-4o",
        #     input="Hello!",
        # )
        #
        # print(response.output_text)

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        #print(response)
        return response.choices[0].message.content


# Example usage
advisor = FinancialAdvisorAgent("")

user_profile = {
    "age": 42,
    "income": 120000,
    "savings": 180000,
    "debt": 220000,  # Mortgage
    "dependents": 2,
    "existing_investments": {
        "stocks": 50000,
        "bonds": 30000,
        "retirement_accounts": 210000
    },
    "risk_tolerance": "moderate"
}


investment_goals = """
I want to save for my children's college education (ages 8 and 10) while also 
growing my retirement fund. I'm concerned about market volatility but want to 
balance growth with reasonable risk. I can invest $1,500 monthly.
"""

advice = advisor.provide_investment_advice(user_profile, investment_goals)
print(advice)
