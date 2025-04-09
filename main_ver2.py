import os
import json
import re
from typing import List
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Flow, start, listen, LLM

class WritingState(BaseModel):
    topic: str = ""
    plan_iterations: List[str] = []
    polished_plan: str = ""
    executed_outline: str = ""
    evaluation: str = ""

class WritingOutlineFlow(Flow[WritingState]):

    @start()
    def get_topic(self):
        print("\n=== Outline Writing Flow ===\n")
        self.state.topic = "Explainable AI in Healthcare"  # Example topic 
        # self.state.topic = input("Enter your writing task/topic: ").strip()
        return self.state

    @listen(get_topic)
    def planner_phase(self, state):
        print("\nPlanner Agent: Generating structured writing strategy...\n")
        llm = LLM(model="ollama/llama3")
        planner = Agent(
            role="Planner",
            goal="Analyze the writing task, generate a problem definition, design plan, and list resources.",
            backstory="Expert in structuring writing strategies and task breakdowns.",
            llm=llm,
            verbose=True
        )

        plan_prompt = f"""
        You are tasked with planning a writing task based on the following topic:

        [TOPIC]: {state.topic}

        Step 1: Understand and describe the overall writing task as a problem statement.
        Step 2: Define the following in JSON format:
        {{
            "Problem Definition": ..., 
            "Goals": ..., 
            "Resources": ..., 
            "Rough Plan": ...,  # this is like a rough outline for the outline writing task
            "Success Metric": ...
        }}

        Revise your plan 3 times and then return a final polished plan.
        """

        task = Task(
            description=plan_prompt,
            agent=planner,
            expected_output="Polished JSON plan",
            output_key="plan_result"
        )

        crew = Crew(agents=[planner], tasks=[task], verbose=True)
        result = crew.kickoff()
        state.polished_plan = result.raw
        state.plan_iterations = result.history
        return state

    @listen(planner_phase)
    def executor_phase(self, state):
        print("\nExecutor Agent: Executing and refining the plan...\n")
        llm = LLM(model="ollama/llama3")
        executor = Agent(
            role="Executor",
            goal="Improve and refine the rough plan into a full outline.",
            backstory="Specialist in turning structured plans into real, coherent outputs.",
            llm=llm,
            verbose=True
        )

        exec_prompt = f"""
        You are the executor. You received this plan:
        {state.polished_plan}

        Conduct 5 rounds of refinement on the "Rough Plan" and generate a complete, structured outline.
        Build iteratively and output the final, polished outline.
        """

        task = Task(
            description=exec_prompt,
            agent=executor,
            expected_output="Final outline",
            output_key="executed_outline"
        )

        crew = Crew(agents=[executor], tasks=[task], verbose=True)
        result = crew.kickoff()
        state.executed_outline = result.raw
        return state

    @listen(executor_phase)
    def evaluator_phase(self, state):
        print("\n🧪 Evaluator Agent: Evaluating final outline against success metrics...\n")
        llm = LLM(model="ollama/llama3")
        evaluator = Agent(
            role="Evaluator",
            goal="Compare the output with the original plan using defined success metrics.",
            backstory="An objective evaluator trained in analyzing plan execution results.",
            llm=llm,
            verbose=True
        )

        eval_prompt = f"""
        Compare the following final output against the original plan.

        Original Plan:
        {state.polished_plan}

        Final Output:
        {state.executed_outline}

        Evaluate how well the output satisfies the Problem Definition, Goals, and Success Metric.
        Provide a score and reasoning. Then suggest potential improvements.
        """

        task = Task(
            description=eval_prompt,
            agent=evaluator,
            expected_output="Evaluation report",
            output_key="evaluation"
        )

        crew = Crew(agents=[evaluator], tasks=[task], verbose=True)
        result = crew.kickoff()
        state.evaluation = result.raw
        return state

    @listen(evaluator_phase)
    def save_results(self, state):
        print("\n💾 Saving all results to output/writing_outline_results.json\n")
        os.makedirs("output", exist_ok=True)
        with open("output/writing_outline_results.json", "w") as f:
            json.dump({
                "topic": state.topic,
                "plan_iterations": state.plan_iterations,
                "polished_plan": state.polished_plan,
                "executed_outline": state.executed_outline,
                "evaluation": state.evaluation
            }, f, indent=2)
        return "✅ Done"

def kickoff():
    WritingOutlineFlow().kickoff()
    print("\n✅ Flow execution complete!")

def plot():
    flow = WritingOutlineFlow()
    flow.plot("writing_outline_flow.html")
    print("Flow diagram saved to writing_outline_flow.html")

if __name__ == "__main__":
    kickoff()