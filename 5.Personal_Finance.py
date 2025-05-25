import openai
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import data_info


class PersonalFinanceAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        self.state = {}

    def create_budget_plan(self, income, expenses, savings_goal, timeline_months):
        """Create a comprehensive budget plan using planning and reasoning."""

        # Organize the expenses data for the model
        expenses_summary = self._summarize_expenses(expenses)

        prompt = f"""
        Create a detailed budget plan based on the following information:

        MONTHLY INCOME: ${income}

        CURRENT EXPENSES:
        {expenses_summary}

        SAVINGS GOAL: ${savings_goal} in {timeline_months} months

        Think through this step-by-step:
        1. Calculate the monthly savings required to reach the goal
        2. Analyze current spending patterns to identify areas for reduction
        3. Create a recommended monthly budget with specific allocations
        4. Suggest concrete actions to reduce expenses in key categories
        5. Develop a contingency plan for unexpected expenses

        Provide the complete budget plan with clear category allocations and actionable recommendations.
        """
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        budget_plan =  response.choices[0].message.content

        # Store the budget plan in agent's state
        self.state["budget_plan"] = budget_plan
        self.state["income"] = income
        self.state["original_expenses"] = expenses
        self.state["savings_goal"] = savings_goal
        self.state["timeline_months"] = timeline_months

        return budget_plan

    def evaluate_purchase_decision(self, purchase_amount, purchase_category, purchase_description):
        """Evaluate whether a purchase aligns with the budget plan."""

        if "budget_plan" not in self.state:
            return "No budget plan exists. Please create a budget plan first."

        prompt = f"""
        Evaluate if this purchase decision aligns with the user's budget plan:

        PROPOSED PURCHASE:
        - Amount: ${purchase_amount}
        - Category: {purchase_category}
        - Description: {purchase_description}

        USER'S BUDGET CONTEXT:
        - Monthly Income: ${self.state["income"]}
        - Savings Goal: ${self.state["savings_goal"]} in {self.state["timeline_months"]} months

        CURRENT BUDGET PLAN:
        {self.state["budget_plan"]}

        Think through this decision step-by-step:
        1. Identify which budget category this purchase falls under
        2. Determine if this purchase exceeds the allocated amount for that category
        3. Assess the necessity and value of this purchase
        4. Consider alternatives or postponement options
        5. Provide a clear recommendation with justification

        Should the user proceed with this purchase? Why or why not?
        """
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        return response.choices[0].message.content

    def _summarize_expenses(self, expenses):
        """Format expenses for better readability in prompts."""
        summary = ""
        for category, amount in expenses.items():
            summary += f"- {category}: ${amount}\n"
        return summary


# Example usage
import data_info
finance_agent = PersonalFinanceAgent(data_info.open_ai_key)

# Create a budget plan
income = 5800
expenses = {
    "Housing": 1800,
    "Utilities": 350,
    "Groceries": 600,
    "Transportation": 400,
    "Entertainment": 450,
    "Dining Out": 500,
    "Shopping": 600,
    "Subscriptions": 120,
    "Miscellaneous": 300
}
savings_goal = 12000
timeline_months = 12

budget_plan = finance_agent.create_budget_plan(income, expenses, savings_goal, timeline_months)
print("BUDGET PLAN:\n", budget_plan)

# Evaluate a purchase decision
purchase_evaluation = finance_agent.evaluate_purchase_decision(
    purchase_amount=899,
    purchase_category="Electronics",
    purchase_description="New smartphone to replace 3-year old device with cracked screen"
)
print("\nPURCHASE EVALUATION:\n", purchase_evaluation)
