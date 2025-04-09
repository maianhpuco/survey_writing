#!/usr/bin/env python
import json
import os
from pydantic import BaseModel
from crewai import LLM, Agent, Task, Crew
from crewai.flow.flow import Flow, listen, start

class WritingState(BaseModel):
    topic: str = ""
    polished_plan: str = ""  # Final JSON string after revisions
    plan_iterations: list = []  # Store intermediate plans for tracking

class WritingOutlineFlow(Flow[WritingState]):
    def __init__(self):
        super().__init__(initial_state=WritingState())

    @start()
    def get_topic(self):
        print("\n=== Outline Writing Flow ===\n")
        self.state.topic = "Explainable AI in Healthcare"  # Example topic
        # self.state.topic = input("Enter your writing task/topic: ").strip()
        return self.state

    @listen("get_topic")
    def planner_phase(self, state: WritingState) -> WritingState:
        print("\nPlanner Agent: Generating structured writing strategy...\n")
        llm = LLM(model="ollama/qwq")
        planner = Agent(
            role="Planner",
            goal="Analyze the writing task, generate a problem definition, design plan, and list resources.",
            backstory="Expert in structuring writing strategies and task breakdowns. Good at planning and critical thinking",
            llm=llm,
            verbose=True
        )

        # Initial prompt for the first iteration
        initial_prompt = f"""
        You are tasked with planning to write the outline a writing task based on the following topic:

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

        This is the first draft. Generate the initial plan.
        """

        # Store iterations
        iterations = []
        
        # First iteration
        task = Task(
            description=initial_prompt,
            agent=planner,
            expected_output="JSON string",
            name="plan_iteration_0"
        )
        crew = Crew(agents=[planner], tasks=[task], verbose=True)
        result = crew.kickoff()
        iterations.append(result.tasks_output[0].raw)

        # Two more rounds of refinement (total 3 iterations)
        for i in range(2):
            previous_plan = iterations[-1]
            refine_prompt = f"""
            Review and refine the previous plan for the writing task on '{state.topic}':

            Previous Plan:
            {previous_plan}

            Improve the plan by:
            - Enhancing clarity and specificity in the Problem Definition
            - Adding detail to Goals and Success Metric
            - Refining the Rough Plan with more structure
            - Updating Resources if needed

            Return the revised plan in JSON format.
            """
            task = Task(
                description=refine_prompt,
                agent=planner,
                expected_output="JSON string",
                name=f"plan_iteration_{i+1}"
            )
            crew = Crew(agents=[planner], tasks=[task], verbose=True)
            result = crew.kickoff()
            iterations.append(result.tasks_output[0].raw)

        # Final polish
        final_prompt = f"""
        You have completed 3 iterations of planning for '{state.topic}'. Here are all iterations:

        Iteration 1: {iterations[0]}
        Iteration 2: {iterations[1]}
        Iteration 3: {iterations[2]}

        Combine the best elements from all iterations into a final polished plan in JSON format:
        {{
            "Problem Definition": "...",
            "Goals": "...",
            "Resources": "...",
            "Rough Plan": "...",
            "Success Metric": "..."
        }}
        """
        task = Task(
            description=final_prompt,
            agent=planner,
            expected_output="Polished JSON plan",
            name="final_plan"
        )
        crew = Crew(agents=[planner], tasks=[task], verbose=True)
        result = crew.kickoff()

        # Store results in state
        state.polished_plan = result.tasks_output[0].raw
        state.plan_iterations = iterations  # Store all intermediate plans
        return state

    @listen("planner_phase")
    def save_output(self, state: WritingState) -> str:
        print("\nSaving planning results to output/writing_plan.json\n")
        os.makedirs("output", exist_ok=True)
        output_file = "output/writing_plan_ver3.json"
        
        # Prepare the data to save
        output_data = {
            "topic": state.topic,
            "polished_plan": json.loads(state.polished_plan),  # Convert JSON string to dict for readability
            "plan_iterations": [json.loads(itr) for itr in state.plan_iterations]  # Convert each iteration to dict
        }
        
        # Write to file
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        return "Done"

def kickoff():
    flow = WritingOutlineFlow()
    flow.kickoff()
    print("\n✅ Outline planning flow complete!")

if __name__ == "__main__":
    kickoff()