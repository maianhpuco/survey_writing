import os
import json
import re
from typing import List
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Flow, start, listen, LLM

from src.database import database
from src.utils import tokenCounter
from src.prompt import ROUGH_OUTLINE_PROMPT, MERGING_OUTLINE_PROMPT, SUBSECTION_OUTLINE_PROMPT, EDIT_FINAL_OUTLINE_PROMPT

class SurveyState(BaseModel):
    topic: str = ""
    titles: List[str] = []
    abstracts: List[str] = []
    abs_chunks: List[List[str]] = []
    title_chunks: List[List[str]] = []
    rough_outlines: List[str] = []
    merged_outline: str = ""
    subsection_outlines: List[str] = []
    final_outline: str = ""

class WriteSurveyOutlineFlow(Flow[SurveyState]):

    @start()
    def get_topic(self):
        print("\n=== Survey Outline Generation Flow ===\n")
        self.state.topic = input("Enter your research topic: ").strip()
        return self.state

    @listen(get_topic)
    def retrieve_references(self, state):
        print("\n📚 Retrieving papers from database...\n")
        db = database()
        refs = db.get_paper_info_from_ids(db.get_ids_from_query(state.topic, num=500))
        state.abstracts = [r['abs'] for r in refs]
        state.titles = [r['title'] for r in refs]
        return state

    @listen(retrieve_references)
    def chunk_references(self, state):
        print("\n🔗 Chunking references...\n")
        token_counter = tokenCounter()
        total_length = token_counter.num_tokens_from_list_string(state.abstracts)
        chunk_size = 14000
        num_chunks = int(total_length / chunk_size) + 1
        avg_len = int(total_length / num_chunks) + 1

        abs_chunks, title_chunks, l, start = [], [], 0, 0
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

    @listen(chunk_references)
    def generate_rough_outlines(self, state):
        print("\n✏️ Generating rough outlines...\n")
        llm = LLM(model="ollama/qwq")
        agent = Agent(
            role="Outline Planner",
            goal="Generate structured survey outlines",
            backstory="You specialize in organizing academic surveys based on related papers.",
            llm=llm,
            verbose=True
        )

        tasks = []
        for i in range(len(state.abs_chunks)):
            paper_texts = ''.join(f"---\npaper_title: {t}\n\npaper_content:\n{p}\n" for t, p in zip(state.title_chunks[i], state.abs_chunks[i]))
            prompt = ROUGH_OUTLINE_PROMPT.replace("[PAPER LIST]", paper_texts).replace("[TOPIC]", state.topic).replace("[SECTION NUM]", "6")
            tasks.append(Task(description=prompt, agent=agent, expected_output="Outline", output_key=f"outline_{i}"))

        crew = Crew(agents=[agent], tasks=tasks, verbose=True)
        result = crew.kickoff()
        state.rough_outlines = list(result.values())
        return state

    @listen(generate_rough_outlines)
    def merge_outlines(self, state):
        print("\n🔀 Merging outlines...\n")
        llm = LLM(model="ollama/qwq")
        agent = Agent(
            role="Outline Merger",
            goal="Combine multiple outlines into one coherent survey outline",
            backstory="You merge outlines into a unified academic survey structure.",
            llm=llm,
            verbose=True
        )
        outline_texts = ''.join(f"---\nOutline {i}:\n{o}\n" for i, o in enumerate(state.rough_outlines))
        prompt = MERGING_OUTLINE_PROMPT.replace("[OUTLINE LIST]", outline_texts).replace("[TOPIC]", state.topic)
        task = Task(description=prompt, agent=agent, expected_output="Merged outline")
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()
        state.merged_outline = result.raw
        return state

    @listen(merge_outlines)
    def finalize_outline(self, state):
        print("\n🧽 Finalizing and cleaning the outline...\n")
        llm = LLM(model="ollama/qwq")
        agent = Agent(
            role="Survey Editor",
            goal="Polish the academic survey outline",
            backstory="An expert at making outlines coherent, complete, and well-structured.",
            llm=llm,
            verbose=True
        )
        prompt = EDIT_FINAL_OUTLINE_PROMPT.replace("[OVERALL OUTLINE]", state.merged_outline)
        task = Task(description=prompt, agent=agent, expected_output="Final outline")
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()
        state.final_outline = result.raw
        return state

    @listen(finalize_outline)
    def save_output(self, state):
        print("\n💾 Saving final outline to output/final_outline.json\n")
        os.makedirs("output", exist_ok=True)
        with open("output/final_outline.json", "w") as f:
            json.dump({
                "topic": state.topic,
                "final_outline": state.final_outline
            }, f, indent=2)
        return "Done"

def kickoff():
    WriteSurveyOutlineFlow().kickoff()
    print("\n✅ Outline flow execution complete!")

def plot():
    flow = WriteSurveyOutlineFlow()
    flow.plot("write_survey_outline_flow.html")
    print("Flow plot saved to write_survey_outline_flow.html")

if __name__ == "__main__":
    kickoff()
  