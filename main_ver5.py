#!/usr/bin/env python
import json
import os
from pydantic import BaseModel
from crewai import LLM, Agent, Task, Crew
from crewai.flow.flow import Flow, listen, start
from src.database import database  # Assuming this is your database module

FIRST_PLANNER_PROMPT = f"""
            You are tasked with planning a writing task based on the following topic and resources:

            [TOPIC]: {state.topic}
            [TASK]: {state.task}
            [SAMPLE REFERENCES FROM CHUNKS]:
            {sample_chunk_refs}

            Define the following in JSON format:
            {{
                "Problem Statement": "Describe the task they are required to solve based on the topic '{state.topic}' and task '{state.task}'",
                "Goals": "Based on the topic '{state.topic}' and task '{state.task}', decide the desirable goals",
                "Resources": "List of resources including the chunked references from state.abs_chunks and state.title_chunks",
                "Initial Result": "Based on the goals and resources, decide the initial result for the survey paper outline",
                "Success Metric": "Define how success will be measured for this task: '{state.task}'"
            }}
            Ensure the output is a valid JSON string enclosed in ```json ... ``` markers.
            """ 

NEXT_PLANNER_PROMPT = f""" 
""" 

EXECUTOR_PROMPT = f"""
""" 
class WritingState(BaseModel):
    topic: str = ""
    abstracts: list = []  # Store paper abstracts
    titles: list = []  # Store paper titles
    abs_chunks: list = []  # Store chunked abstracts
    title_chunks: list = []  # Store chunked titles
    rough_outlines: list = []  # Store rough outlines from chunks
    polished_plan: str = ""  # Final JSON string after Executor revisions
    plan_iterations: list = []  # Store intermediate plans from Executor
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
        self.state.task = 'Writing the outline for a Survey paper with the topic: {}'.format(self.state.topic)
        return self.state

    @listen("get_topic")
    def retrieve_references(self, state: WritingState) -> WritingState:
        print("\nRetrieving papers from database...\n")
        db = database(
            db_path="/project/hnguyen2/mvu9/datasets/llms/auto_survey/database",
            embedding_model="nomic-ai/nomic-embed-text-v1"
        )
        # Refined query for better relevance
        query = "Explainable AI Healthcare"  # More specific than just the topic
        refs = db.get_paper_info_from_ids(db.get_ids_from_query(query, num=50))
        state.abstracts = [r['abs'] for r in refs]
        state.titles = [r['title'] for r in refs]
        return state 

    @listen("retrieve_references")
    def chunk_references(self, state: WritingState) -> WritingState:
        print("\nChunking references...\n")
        token_counter = tokenCounter()
        total_length = token_counter.num_tokens_from_list_string(state.abstracts)
        chunk_size = 14000
        num_chunks = int(total_length / chunk_size) + 1
        avg_len = int(total_length / num_chunks) + 1

        abs_chunks, title_chunks = [], []
        l, start = 0, 0
        for i in range(len(state.abstracts)):
            l += token_counter.num_tokens_from_string(state.abstracts[i])
            if l > avg_len:
                abs_chunks.append(state.abstracts[start:i])
                title_chunks.append(state.titles[start:i])
                start = i
                l = 0
        abs_chunks.append(state.abstracts[start:])
        title_chunks.append(state.titles[start:])

        state.abs_chunks = abs_chunks
        state.title_chunks = title_chunks
        return state
    
    @listen("chunk_references")
    def planner_phase(self, state: WritingState) -> WritingState:
        print(f"\nPlanner Agent: Generating initial writing strategy (Round {state.planner_rounds + 1})...\n")
        llm = LLM(model="ollama/qwq")
        planner = Agent(
            role="Planner",
            goal="Analyze the writing task and resources to generate an initial plan.",
            backstory="Expert in structuring writing strategies and task breakdowns.",
            llm=llm,
            verbose=True
        )

        # Use a sample of chunked references (e.g., first chunk) for the initial plan
        sample_chunk_refs = "\n".join(
            [f"Title: {t}\nAbstract: {a}" for t, a in zip(state.title_chunks[0][:3], state.abs_chunks[0][:3])]
        ) if state.abs_chunks and state.title_chunks else "No chunked references available."

        if state.planner_rounds == 0:
            plan_prompt = FIRST_PLANNER_PROMPT
        else:
            plan_prompt = f"""
            Revise your previous plan based on this evaluation feedback:

            Previous Plan:
            {state.polished_plan}

            Evaluation Feedback:
            {json.dumps(state.evaluation_result, indent=2)}

            [TOPIC]: {state.topic}
            [TASK]: {state.task}
            [SAMPLE REFERENCES FROM CHUNKS]:
            {sample_chunk_refs}

            Improve the plan by addressing the feedback. Return a revised plan in JSON format:
            {{
                "Problem Statement": "Describe the task they are required to solve based on the topic '{state.topic}' and task '{state.task}'",
                "Goals": "Based on the topic '{state.topic}' and task '{state.task}', decide the desirable goals",
                "Resources": "List of resources including the chunked references from state.abs_chunks and state.title_chunks",
                "Rough Plan": "Based on the goals and resources, decide the revised result for the survey paper outline",
                "Success Metric": "Define how success will be measured for this task: '{state.task}'"
            }}
            Ensure the output is a valid JSON string enclosed in ```json ... ``` markers.
            """

        task = Task(
            description=plan_prompt,
            agent=planner,
            expected_output="JSON string",
            name="initial_plan"
        )
        crew = Crew(agents=[planner], tasks=[task], verbose=True)
        result = crew.kickoff()
        raw_output = result.tasks_output[0].raw.strip()
        if raw_output.startswith("```json") and raw_output.endswith("```"):
            raw_output = raw_output[7:-3].strip()

        try:
            json.loads(raw_output)
            state.polished_plan = raw_output
        except json.JSONDecodeError as e:
            print(f"Error: Initial plan is not valid JSON: {raw_output}")
            state.polished_plan = '{"error": "Failed to generate valid JSON plan"}'

        return state
    
    @listen("planner_phase")
    def executor_phase(self, state: WritingState) -> WritingState:
        print("\nExecutor Agent: Writing and refining the survey paper outline in 5 rounds...\n")
        llm = LLM(model="ollama/qwq")
        executor = Agent(
            role="Executor",
            goal="Execute the task of writing a survey paper outline and refine it iteratively.",
            backstory="Specialist in executing and polishing writing tasks into detailed outlines.",
            llm=llm,
            verbose=True
        )

        # Start with the Planner's initial result
        initial_plan = json.loads(state.polished_plan).get("Initial Result", "I. Introduction\nII. Body\nIII. Conclusion")
        iterations = [initial_plan]

        # Use a sample of chunked references for context
        sample_chunk_refs = "\n".join(
            [f"Title: {t}\nAbstract: {a}" for t, a in zip(state.title_chunks[0][:3], state.abs_chunks[0][:3])]
        ) if state.abs_chunks and state.title_chunks else "No chunked references available."

        # 5-round execution and refinement loop
        for i in range(5):
            refine_prompt = f"""
            Improve this survey paper outline based on the previous result, topic, task, and resources:

            Previous Outline:
            {iterations[-1]}

            Topic: {state.topic}
            Task: {state.task}
            Sample References (from chunks):
            {sample_chunk_refs}

            Execute the task of writing the survey paper outline by refining the previous result. Enhance detail, structure, and relevance to the topic and task. Return the improved outline as a JSON string with an "Outline" key:
            {{
                "Outline": "Improved survey paper outline text"
            }}
            Ensure the output is enclosed in ```json ... ``` markers.
            """
            task = Task(
                description=refine_prompt,
                agent=executor,
                expected_output="JSON string",
                name=f"outline_iteration_{i}"
            )
            crew = Crew(agents=[executor], tasks=[task], verbose=True)
            result = crew.kickoff()
            raw_output = result.tasks_output[0].raw.strip()
            if raw_output.startswith("```json") and raw_output.endswith("```"):
                raw_output = raw_output[7:-3].strip()
            try:
                draft_json = json.loads(raw_output)
                iterations.append(draft_json["Outline"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Invalid JSON in iteration {i}: {raw_output}")
                iterations.append(f"Error in iteration {i}")

        # Combine the 5 rounds into a polished version
        combine_prompt = f"""
        Combine and polish these 5 iterative outlines into a final survey paper outline:

        Iteration 1: {iterations[1]}
        Iteration 2: {iterations[2]}
        Iteration 3: {iterations[3]}
        Iteration 4: {iterations[4]}
        Iteration 5: {iterations[5]}

        Topic: {state.topic}
        Task: {state.task}
        Sample References (from chunks):
        {sample_chunk_refs}

        Synthesize the best elements from each iteration into a cohesive, detailed, and well-structured outline. Return the final outline as a JSON string with an "Outline" key:
        {{
            "Outline": "Final polished survey paper outline text"
        }}
        Ensure the output is enclosed in ```json ... ``` markers.
        """
        task = Task(
            description=combine_prompt,
            agent=executor,
            expected_output="JSON string",
            name="final_outline"
        )
        crew = Crew(agents=[executor], tasks=[task], verbose=True)
        result = crew.kickoff()
        raw_final = result.tasks_output[0].raw.strip()
        if raw_final.startswith("```json") and raw_final.endswith("```"):
            raw_final = raw_final[7:-3].strip()

        # Update polished_plan with the final polished outline
        try:
            final_json = json.loads(raw_final)
            state.polished_plan = json.dumps({
                **json.loads(state.polished_plan),  # Keep original fields
                "Rough Plan": final_json["Outline"]  # Update with final polished outline
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error: Final outline is not valid JSON: {raw_final}")
            state.polished_plan = json.dumps({
                **json.loads(state.polished_plan),
                "Rough Plan": "Error in final outline generation"
            })

        state.plan_iterations = iterations  # Store all 5 drafts plus initial
        return state 

    @listen("executor_phase")
    def evaluator_phase(self, state: WritingState) -> WritingState:
        print("\nEvaluator Agent: Assessing the polished outline against Planner's goals...\n")
        llm = LLM(model="ollama/qwq")
        evaluator = Agent(
            role="Evaluator",
            goal="Evaluate whether the Executor's polished outline meets the Planner's goals.",
            backstory="Expert in assessing writing plans and their execution for quality and alignment with objectives.",
            llm=llm,
            verbose=True
        )

        # Extract Planner's original plan and Executor's polished version
        planner_plan = json.loads(state.polished_plan)  # Contains both initial and final fields
        planner_goals = planner_plan.get("Goals", "No goals defined")
        planner_success_metric = planner_plan.get("Success Metric", "No success metric defined")
        executor_outline = planner_plan.get("Rough Plan", "No polished outline available")

        eval_prompt = f"""
        As an Evaluator, assess whether the Executor's polished survey paper outline meets the Planner's goals:

        Planner's Goals:
        {planner_goals}

        Planner's Success Metric:
        {planner_success_metric}

        Executor's Polished Outline:
        {executor_outline}

        Evaluate the polished outline based on:
        - Alignment with Goals: Does it fulfill the objectives set by the Planner?
        - Completeness: Does it cover all necessary aspects as per the goals?
        - Clarity: Is the outline well-structured and clear?
        - Relevance: Does it align with the topic '{state.topic}' and task '{state.task}'?

        Return a JSON evaluation:
        {{
            "alignment_with_goals": "comment on how well it meets the Planner's goals",
            "completeness": "comment on coverage of necessary aspects",
            "clarity": "comment on structure and readability",
            "relevance": "comment on alignment with topic and task",
            "score": number (0-100),
            "recommendations": "suggestions for improvement if needed"
        }}
        Ensure the output is a valid JSON string enclosed in ```json ... ``` markers.
        """
        task = Task(
            description=eval_prompt,
            agent=evaluator,
            expected_output="JSON string",
            name="evaluate_outline"
        )
        crew = Crew(agents=[evaluator], tasks=[task], verbose=True)
        result = crew.kickoff()
        raw_eval = result.tasks_output[0].raw.strip()
        if raw_eval.startswith("```json") and raw_eval.endswith("```"):
            raw_eval = raw_eval[7:-3].strip()

        # Store evaluation result
        try:
            state.evaluation_result = json.loads(raw_eval)
        except json.JSONDecodeError as e:
            print(f"Error: Evaluation result is not valid JSON: {raw_eval}")
            state.evaluation_result = {
                "error": "Invalid JSON from Evaluator",
                "score": 0,
                "recommendations": "Retry evaluation due to output error"
            }

        # Check score and decide next step
        score = state.evaluation_result.get("score", 0)
        threshold = 80  # Threshold set to 80%
        if score < threshold and state.planner_rounds < self.max_planner_rounds:
            print(f"\n---Evaluation score {score} < {threshold}. Returning to Planner for round {state.planner_rounds + 2}...\n")
            state.planner_rounds += 1  # Increment before looping back
            return self.planner_phase(state)
        else:
            print(f"\nEvaluation complete. Score: {score}. Proceeding to save output...\n")
            return state 

    @listen("evaluator_phase")
    def save_output(self, state: WritingState) -> str:
        print("\n💾 Saving all state outputs and final result to output/writing_outline.json\n")
        os.makedirs("output", exist_ok=True)
        output_file = "output/writing_outline_ver5.json"

        # Prepare polished_plan, handling potential invalid JSON
        try:
            polished_plan_dict = json.loads(state.polished_plan)
        except json.JSONDecodeError:
            polished_plan_dict = {"error": "Invalid JSON in polished_plan", "raw": state.polished_plan}

        # Prepare plan_iterations, handling potential invalid JSON
        try:
            plan_iterations_list = [
                json.loads(itr) if itr.startswith("{") else itr 
                for itr in state.plan_iterations
            ]
        except json.JSONDecodeError:
            plan_iterations_list = [
                {"error": "Invalid JSON in iteration", "raw": itr} 
                if itr.startswith("{") else itr 
                for itr in state.plan_iterations
            ]

        # Structure the output data
        output_data = {
            "topic": state.topic,
            "task": state.task,
            "abstracts_sample": state.abstracts[:5],  # Sample of 5 for brevity
            "titles_sample": state.titles[:5],  # Sample of 5 for brevity
            "rough_outlines": state.rough_outlines,  # Empty in this version, included for completeness
            "polished_plan": polished_plan_dict,  # Final result from Executor
            "plan_iterations": plan_iterations_list,  # All 5 Executor iterations plus initial
            "evaluation_result": state.evaluation_result,  # Evaluator's assessment
            "planner_rounds": state.planner_rounds  # Number of planning iterations
        }

        # Save to file
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        return "Done"  
    

def kickoff():
    flow = WritingOutlineFlow()
    flow.kickoff()
    print("\n✅ Outline planning flow complete!")

if __name__ == "__main__":
    kickoff() 