from crewai import BaseState
from typing import List

class SurveyState(BaseState):
    topic: str = ""
    titles: List[str] = []
    abstracts: List[str] = []
    abs_chunks: List[List[str]] = []
    title_chunks: List[List[str]] = []
    rough_outlines: List[str] = []
    merged_outline: str = ""
    subsection_outlines: List[str] = []
    final_outline: str = ""

import os
import json
import re
from crewai import Agent, Task, Crew, Flow, start, listen
from your_module.database import database
from your_module.utils import tokenCounter
from your_module.prompts import ROUGH_OUTLINE_PROMPT, MERGING_OUTLINE_PROMPT, SUBSECTION_OUTLINE_PROMPT, EDIT_FINAL_OUTLINE_PROMPT
import ollama

class WriteSurveyOutlineFlow(Flow[SurveyState]):

    @start()
    def get_topic(self):
        print("\n=== Survey Outline Generation Flow ===\n")
        self.state.topic = input("Enter your research topic: ").strip()
        return self.state

    @listen(get_topic)
    def retrieve_references(self, state):
        print("\n📚 Retrieving papers...\n")
        db = database(db_path="/project/hnguyen2/mvu9/datasets/llms/auto_survey/database")
        refs = db.get_paper_info_from_ids(
            db.get_ids_from_query(state.topic, num=600))

        state.abstracts = [r['abs'] for r in refs]
        state.titles = [r['title'] for r in refs]
        return state

    # You’ll continue defining methods like:
    # - chunking
    # - generate_rough_outlines
    # - merge_outlines
    # - generate_subsections
    # - edit_outline
    # - save_result

    # Each using agents and Crew like in your example

    @listen(retrieve_references)
    def generate_rough_outlines(self, state):
        print("\n✏️ Generating rough outlines...\n")

        # Chunk abstracts
        token_counter = tokenCounter()
        total_tokens = token_counter.num_tokens_from_list_string(state.abstracts)
        avg_tokens = int(total_tokens / 5) + 1
        abs_chunks, title_chunks = [], []
        l, start = 0, 0
        for i, abstract in enumerate(state.abstracts):
            l += token_counter.num_tokens_from_string(abstract)
            if l > avg_tokens:
                abs_chunks.append(state.abstracts[start:i])
                title_chunks.append(state.titles[start:i])
                l = 0
                start = i
        abs_chunks.append(state.abstracts[start:])
        title_chunks.append(state.titles[start:])
        state.abs_chunks = abs_chunks
        state.title_chunks = title_chunks

        # LLM agent setup
        agent = Agent(
            role="Outline Planner",
            goal="Generate structured survey outlines from paper abstracts",
            backstory="An AI trained on thousands of academic surveys.",
            llm=LLM(model="ollama/qwq"),
            verbose=True
        )

        tasks = []
        for i in range(len(abs_chunks)):
            paper_list = ''.join(f'---\npaper_title: {t}\n\npaper_content:\n{p}\n' for t, p in zip(title_chunks[i], abs_chunks[i]))
            prompt = ROUGH_OUTLINE_PROMPT.replace("[PAPER LIST]", paper_list).replace("[TOPIC]", state.topic).replace("[SECTION NUM]", "6")

            task = Task(
                description="Generate a rough outline",
                agent=agent,
                prompt=prompt,
                expected_output="One outline in format Title + Section + Description",
                output_key=f"outline_{i}"
            )
            tasks.append(task)

        crew = Crew(agents=[agent], tasks=tasks, verbose=True)
        result = crew.kickoff()

        state.rough_outlines = list(result.values())
        return state
    