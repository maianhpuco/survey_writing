#!/usr/bin/env python
import json
import os 
from pydantic import BaseModel
from crewai import LLM, Agent, Task, Crew
from crewai.flow.flow import Flow, listen, start

class WritingState(BaseModel):
    topic: str = ""
    polished_plan: str = ""  # Final JSON string after revisions
    plan_iterations: list = []  # Store intermediate plans from Planner
    evaluation_result: dict = {}  # Store evaluation output
    planner_rounds: int = 0  # Track number of Planner-Evaluator loops

class WritingOutlineFlow(Flow[WritingState]):
    def __init__(self):
        super().__init__(initial_state=WritingState())
        self.max_planner_rounds = 3  # Maximum number of Planner-Evaluator loops

    @start()
    def get_topic(self):
        print("\n=== Outline Writing Flow ===\n")
        self.state.topic = "Explainable AI in Healthcare"  # Example topic
        return self.state

    @listen("get_topic")
    def planner_phase(self, state: WritingState) -> WritingState:
        print(f"\nPlanner Agent: Generating structured writing strategy (Round {state.planner_rounds + 1})...\n")
        llm = LLM(model="ollama/llama3")
        planner = Agent(
            role="Planner",
            goal="Analyze the writing task, generate a problem definition, design plan, and list resources.",
            backstory="Expert in structuring writing strategies and task breakdowns.",
            llm=llm,
            verbose=True
        )

        # Initial prompt for first round, or feedback-based prompt for subsequent rounds
        if state.planner_rounds == 0:
            plan_prompt = f"""
            You are tasked with planning a writing task based on the following topic:

            [TOPIC]: {state.topic}

            Step 1: Understand and describe the overall writing task as a problem statement.
            Step 2: Define the following in JSON format:
            {{
                "Problem Definition": "Describe the writing task as a problem to solve",
                "Goals": "List the objectives of the writing task",
                "Resources": "List available or needed resources",
                "Rough Plan": "Provide a rough outline or plan for the writing task",
                "Success Metric": "Define how success will be measured"
            }}

            Revise your plan 3 times internally and return a polished JSON plan.
            """
        else:
            # Use feedback from Evaluator
            plan_prompt = f"""
            Revise your previous plan based on this evaluation feedback:

            Previous Plan:
            {state.polished_plan}

            Evaluation Feedback:
            {json.dumps(state.evaluation_result, indent=2)}

            Improve the plan by addressing the feedback. Revise it 3 times internally and return a polished JSON plan in this format:
            {{
                "Problem Definition": "...",
                "Goals": "...",
                "Resources": "...",
                "Rough Plan": "...",
                "Success Metric": "..."
            }}
            """

        # Internal 3-iteration refinement loop
        iterations = []
        current_prompt = plan_prompt
        for i in range(3):
            task = Task(
                description=current_prompt,
                agent=planner,
                expected_output="JSON string",
                name=f"plan_iteration_{i}"
            )
            crew = Crew(agents=[planner], tasks=[task], verbose=True)
            result = crew.kickoff()
            iterations.append(result.tasks_output[0].raw)
            # Update prompt for next iteration to refine the previous output
            current_prompt = f"""
            Refine this plan further:
            {iterations[-1]}
            Return the improved plan in JSON format.
            """

        # Final polish of internal iterations
        final_prompt = f"""
        Combine these 3 iterations into a polished plan:
        Iteration 1: {iterations[0]}
        Iteration 2: {iterations[1]}
        Iteration 3: {iterations[2]}
        Return the final plan in JSON format.
        """
        task = Task(
            description=final_prompt,
            agent=planner,
            expected_output="Polished JSON plan",
            name="final_plan"
        )
        crew = Crew(agents=[planner], tasks=[task], verbose=True)
        result = crew.kickoff()

        state.polished_plan = result.tasks_output[0].raw
        state.plan_iterations.extend(iterations)  # Append to track all iterations
        state.planner_rounds += 1
        return state

    @listen("planner_phase")
    def evaluator_phase(self, state: WritingState) -> WritingState:
        print("\nEvaluator Agent: Assessing the plan...\n")
        llm = LLM(model="ollama/llama3")
        evaluator = Agent(
            role="Evaluator",
            goal="Evaluate the plan against success criteria",
            backstory="Expert in assessing plans for completeness and quality.",
            llm=llm,
            verbose=True
        )

        eval_prompt = f"""
        Evaluate this plan:
        {state.polished_plan}

        Assess it based on:
        - Completeness: Does it cover all necessary aspects?
        - Clarity: Is it well-structured and clear?
        - Feasibility: Are the resources and plan realistic?

        Return a JSON evaluation:
        {{
            "completeness": "comment",
            "clarity": "comment",
            "feasibility": "comment",
            "score": number (0-100),
            "recommendations": "suggestions for improvement"
        }}
        """
        task = Task(
            description=eval_prompt,
            agent=evaluator,
            expected_output="JSON string",
            name="evaluate_plan"
        )
        crew = Crew(agents=[evaluator], tasks=[task], verbose=True)
        result = crew.kickoff()
        state.evaluation_result = json.loads(result.tasks_output[0].raw)

        # Check if plan passes (e.g., score >= 90) or max rounds reached
        score = state.evaluation_result.get("score", 0)
        if score < 90 and state.planner_rounds < self.max_planner_rounds:
            print(f"\nEvaluation score {score} < 90. Returning to Planner...\n")
            return self.planner_phase(state)  # Loop back to Planner
        else:
            print(f"\nEvaluation complete. Score: {score}. Proceeding...\n")
            return state

    @listen("evaluator_phase")
    def save_output(self, state: WritingState) -> str:
        print("\n💾 Saving final plan and evaluation to output/writing_plan.json\n")
        os.makedirs("output", exist_ok=True)
        with open("output/writing_plan.json", "w") as f:
            json.dump({
                "topic": state.topic,
                "polished_plan": json.loads(state.polished_plan),  # Convert string to dict
                "evaluation_result": state.evaluation_result,
                "planner_rounds": state.planner_rounds,
                "plan_iterations": [json.loads(itr) for itr in state.plan_iterations]  # Convert strings to dicts
            }, f, indent=2)
        return "Done"

def kickoff():
    flow = WritingOutlineFlow()
    flow.kickoff()
    print("\n✅ Outline planning flow complete!")

if __name__ == "__main__":
    kickoff()