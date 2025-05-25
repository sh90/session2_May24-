import openai
import json
import re


class EducationalTutoringAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        self.student_profile = {}
        self.learning_history = []

    def set_student_profile(self, profile):
        """Set or update the student profile."""
        self.student_profile = profile

    def add_learning_interaction(self, interaction):
        """Add a learning interaction to the history."""
        self.learning_history.append(interaction)

    def explain_concept(self, subject, concept, difficulty_level):
        """Explain an educational concept using structured reasoning."""

        # Build context from student profile and learning history
        profile_context = json.dumps(self.student_profile,
                                     indent=2) if self.student_profile else "No student profile available"

        # Include only relevant history for the subject
        relevant_history = [h for h in self.learning_history if h.get("subject") == subject]
        history_context = json.dumps(relevant_history[-3:],
                                     indent=2) if relevant_history else "No prior learning history for this subject"

        prompt = f"""
        Explain the following educational concept, adapting to the student's profile:

        SUBJECT: {subject}
        CONCEPT: {concept}
        DIFFICULTY LEVEL: {difficulty_level} (1-5 scale)

        STUDENT PROFILE:
        {profile_context}

        RELEVANT LEARNING HISTORY:
        {history_context}

        Explain this concept using the following approach:
        1. Start with a simple definition aligned with the student's current knowledge
        2. Provide a real-world analogy that connects to the student's interests or experiences
        3. Break down the concept into clear, sequential components
        4. Explain each component with examples of increasing complexity
        5. Connect the concept to previously learned material
        6. Include 2-3 practice questions that check understanding

        Adapt your explanation to the student's:
        - Learning style (if known)
        - Prior knowledge in this subject
        - Specific areas of interest
        - Any learning challenges mentioned in their profile

        The explanation should be clear, engaging, and matched to the student's abilities.
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        explanation = response.choices[0].message.content

        # Extract practice questions for later reference
        practice_questions = self._extract_practice_questions(explanation)

        # Save this interaction to history
        self.add_learning_interaction({
            "timestamp": datetime.now().isoformat(),
            "subject": subject,
            "concept": concept,
            "difficulty_level": difficulty_level,
            "interaction_type": "concept_explanation",
            "practice_questions": practice_questions
        })

        return {
            "explanation": explanation,
            "practice_questions": practice_questions
        }

    def _extract_practice_questions(self, text):
        """Extract practice questions from text for future reference."""
        # Simple pattern matching - this could be more sophisticated
        questions = []

        # Look for numbered questions
        pattern = r"(?:Question|Q)\.?\s*(\d+)[.:]?\s*(.*?)(?=(?:Question|Q)\.?\s*\d+|$)"
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            for num, content in matches:
                questions.append(content.strip())
        else:
            # If numbered pattern doesn't work, try to find question marks
            question_marks = re.findall(r"(.*?\?)", text)
            for q in question_marks:
                if len(q.split()) > 5:  # Simple filter for actual questions vs. short phrases
                    questions.append(q.strip())

        return questions

    def evaluate_answer(self, question, student_answer, subject):
        """Evaluate a student's answer to a question with detailed feedback."""

        prompt = f"""
        Evaluate this student's answer to a question:

        SUBJECT: {subject}
        QUESTION: {question}
        STUDENT'S ANSWER: {student_answer}

        Provide a detailed evaluation that includes:
        1. Whether the answer is correct, partially correct, or incorrect
        2. Specific strengths in the student's understanding
        3. Any misconceptions or errors that need addressing
        4. The correct answer or solution process (if the student's answer is incorrect)
        5. Suggestions for improving understanding
        6. A follow-up question that helps deepen understanding

        Focus on being constructive and encouraging while giving clear feedback.
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        evaluation = response.choices[0].message.content

        # Add to learning history
        self.add_learning_interaction({
            "timestamp": datetime.now().isoformat(),
            "subject": subject,
            "interaction_type": "answer_evaluation",
            "question": question,
            "student_answer": student_answer,
            "evaluation": evaluation
        })

        return evaluation

    def generate_personalized_exercise(self, subject, concept, difficulty_level, exercise_type="problem_solving"):
        """Generate a personalized exercise for a student based on their profile and history."""

        # Build context
        profile_context = json.dumps(self.student_profile,
                                     indent=2) if self.student_profile else "No student profile available"

        # Get learning history related to this concept
        concept_history = [h for h in self.learning_history
                           if h.get("subject") == subject and h.get("concept") == concept]

        # Determine knowledge level from history
        if concept_history:
            # Analyze prior interactions to estimate knowledge level
            knowledge_context = f"Student has {len(concept_history)} prior interactions with this concept."
        else:
            knowledge_context = "Student has no recorded history with this specific concept."

        prompt = f"""
        Create a personalized {exercise_type} exercise for this student:

        SUBJECT: {subject}
        CONCEPT: {concept}
        DIFFICULTY LEVEL: {difficulty_level} (1-5 scale)
        EXERCISE TYPE: {exercise_type}

        STUDENT PROFILE:
        {profile_context}

        KNOWLEDGE CONTEXT:
        {knowledge_context}

        Design an exercise that:
        1. Targets the core principles of the concept at the appropriate difficulty level
        2. Connects to the student's interests or real-world applications if possible
        3. Builds on their current understanding
        4. Challenges them appropriately without causing frustration
        5. Can be completed in 10-15 minutes

        Include:
        - Clear instructions
        - The exercise itself
        - A detailed solution or rubric
        - Learning objectives for this exercise
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        exercise = response.choices[0].message.content

        # Add to learning history
        self.add_learning_interaction({
            "timestamp": datetime.now().isoformat(),
            "subject": subject,
            "concept": concept,
            "difficulty_level": difficulty_level,
            "interaction_type": "personalized_exercise",
            "exercise_type": exercise_type,
            "exercise_content": exercise
        })

        return exercise


# Example usage
from datetime import datetime
import data_info

tutor = EducationalTutoringAgent(data_info.open_ai_key)

# Set up student profile
student_profile = {
    "name": "Alex Chen",
    "age": 15,
    "grade": 10,
    "learning_style": "visual",
    "interests": ["video games", "basketball", "space exploration"],
    "strengths": ["problem solving", "creativity"],
    "areas_for_growth": ["mathematical formulas", "writing detailed explanations"],
    "preferred_examples": "technology and sports-related"
}

tutor.set_student_profile(student_profile)

# Explain a concept
physics_explanation = tutor.explain_concept(
    subject="Physics",
    concept="Newton's Second Law of Motion (F=ma)",
    difficulty_level=3
)

print("CONCEPT EXPLANATION:\n", physics_explanation["explanation"])
print("\nEXTRACTED PRACTICE QUESTIONS:")
for i, question in enumerate(physics_explanation["practice_questions"]):
    print(f"{i + 1}. {question}")

# Evaluate a student answer
student_answer = """
The Second Law says that force equals mass times acceleration (F=ma). 
So if you push a heavy object and a light object with the same force, 
the light object will accelerate more because it has less mass.
"""

evaluation = tutor.evaluate_answer(
    physics_explanation["practice_questions"][0] if physics_explanation["practice_questions"] else "Explain Newton's Second Law of Motion",
    student_answer,
    "Physics"
)

print("\nANSWER EVALUATION:\n", evaluation)

# Generate a personalized exercise
exercise = tutor.generate_personalized_exercise(
    subject="Physics",
    concept="Newton's Second Law of Motion (F=ma)",
    difficulty_level=3,
    exercise_type="problem_solving"
)

print("\nPERSONALIZED EXERCISE:\n", exercise)
