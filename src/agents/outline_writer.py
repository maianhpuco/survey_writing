import os
import numpy as np
import tiktoken
from tqdm import trange, tqdm
import time
import torch
import ollama
from src.database import database
from src.utils import tokenCounter
from src.prompt import ROUGH_OUTLINE_PROMPT, MERGING_OUTLINE_PROMPT, SUBSECTION_OUTLINE_PROMPT, EDIT_FINAL_OUTLINE_PROMPT
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

class outlineWriter():

    def __init__(self, model=None, ollama_model=None, api_key=None, api_url=None, database=None) -> None:
        self.model = ollama_model if ollama_model is not None else model
        self.db = database
        self.token_counter = tokenCounter()
        self.input_token_usage, self.output_token_usage = 0, 0

    def _chat(self, prompt):
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']

    def _batch_chat(self, prompts, temperature=1):
        responses = []
        for prompt in prompts:
            response = self._chat(prompt)
            responses.append(response)
        return responses

    def draft_outline(self, topic, reference_num=600, chunk_size=30000, section_num=6):
        references_ids = self.db.get_ids_from_query(topic, num=reference_num, shuffle=True)
        references_infos = self.db.get_paper_info_from_ids(references_ids)
        references_titles = [r['title'] for r in references_infos]
        references_abs = [r['abs'] for r in references_infos]
        print("len of reference title and reference abs",len(references_titles), len(references_abs))
        
        abs_chunks, titles_chunks = self.chunking(references_abs, references_titles, chunk_size=chunk_size)
        len(abs_chunks), len(titles_chunks) 
        print("len of abs_chunks and titles_chunks", len(abs_chunks), len(titles_chunks)) 
        outlines = self.generate_rough_outlines(
            topic=topic, papers_chunks=abs_chunks, titles_chunks=titles_chunks, section_num=section_num)
        print("len of outlines", len(outlines))
        print("outlines 1 ", outlines[1])
        print("outlines 2", outlines[2]) 
        section_outline = self.merge_outlines(topic=topic, outlines=outlines)

        subsection_outlines = self.generate_subsection_outlines(topic=topic, section_outline=section_outline, rag_num=50)

        merged_outline = self.process_outlines(section_outline, subsection_outlines)

        final_outline = self.edit_final_outline(merged_outline)

        return final_outline

    def generate_rough_outlines(self, topic, papers_chunks, titles_chunks, section_num=8):
        prompts = []
        for i in trange(len(papers_chunks)):
            titles = titles_chunks[i]
            papers = papers_chunks[i]
            paper_texts = ''
            for t, p in zip(titles, papers):
                paper_texts += f'---\npaper_title: {t}\n\npaper_content:\n\n{p}\n'
            paper_texts += '---\n'
            prompt = self.__generate_prompt(ROUGH_OUTLINE_PROMPT, paras={'PAPER LIST': paper_texts, 'TOPIC': topic, 'SECTION NUM': str(section_num)})
            prompts.append(prompt)

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)
        outlines = self._batch_chat(prompts, temperature=1)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(outlines)
        return outlines

    def merge_outlines(self, topic, outlines):
        outline_texts = ''
        for i, o in enumerate(outlines):
            outline_texts += f'---\noutline_id: {i}\n\noutline_content:\n\n{o}\n'
        outline_texts += '---\n'
        prompt = self.__generate_prompt(MERGING_OUTLINE_PROMPT, paras={'OUTLINE LIST': outline_texts, 'TOPIC': topic})
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
        outline = self._chat(prompt)
        self.output_token_usage += self.token_counter.num_tokens_from_string(outline)
        return outline

    def generate_subsection_outlines(self, topic, section_outline, rag_num):
        survey_title, survey_sections, survey_section_descriptions = self.extract_title_sections_descriptions(section_outline)

        prompts = []
        for section_name, section_description in zip(survey_sections, survey_section_descriptions):
            references_ids = self.db.get_ids_from_query(section_description, num=rag_num, shuffle=True)
            references_infos = self.db.get_paper_info_from_ids(references_ids)

            references_titles = [r['title'] for r in references_infos]
            references_papers = [r['abs'] for r in references_infos]
            paper_texts = ''
            for t, p in zip(references_titles, references_papers):
                paper_texts += f'---\npaper_title: {t}\n\npaper_content:\n\n{p}\n'
            paper_texts += '---\n'
            prompt = self.__generate_prompt(SUBSECTION_OUTLINE_PROMPT, paras={'OVERALL OUTLINE': section_outline, 'SECTION NAME': section_name,
                                                                              'SECTION DESCRIPTION': section_description, 'TOPIC': topic, 'PAPER LIST': paper_texts})
            prompts.append(prompt)

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)
        sub_outlines = self._batch_chat(prompts, temperature=1)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(sub_outlines)
        return sub_outlines

    def edit_final_outline(self, outline):
        prompt = self.__generate_prompt(EDIT_FINAL_OUTLINE_PROMPT, paras={'OVERALL OUTLINE': outline})
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
        outline = self._chat(prompt)
        self.output_token_usage += self.token_counter.num_tokens_from_string(outline)
        return outline.replace('<format>\n', '').replace('</format>', '')

    def __generate_prompt(self, template, paras):
        prompt = template
        for k in paras:
            prompt = prompt.replace(f'[{k}]', paras[k])
        return prompt

    def extract_title_sections_descriptions(self, outline):
        title = outline.split('Title: ')[1].split('\n')[0]
        sections, descriptions = [], []
        for i in range(100):
            if f'Section {i+1}' in outline:
                sections.append(outline.split(f'Section {i+1}: ')[1].split('\n')[0])
                descriptions.append(outline.split(f'Description {i+1}: ')[1].split('\n')[0])
        return title, sections, descriptions

    def extract_subsections_subdescriptions(self, outline):
        subsections, subdescriptions = [], []
        for i in range(100):
            if f'Subsection {i+1}' in outline:
                subsections.append(outline.split(f'Subsection {i+1}: ')[1].split('\n')[0])
                subdescriptions.append(outline.split(f'Description {i+1}: ')[1].split('\n')[0])
        return subsections, subdescriptions

    def chunking(self, papers, titles, chunk_size=14000):
        paper_chunks, title_chunks = [], []
        total_length = self.token_counter.num_tokens_from_list_string(papers)
        num_of_chunks = int(total_length / chunk_size) + 1
        avg_len = int(total_length / num_of_chunks) + 1
        split_points, l = [], 0

        for j in range(len(papers)):
            l += self.token_counter.num_tokens_from_string(papers[j])
            if l > avg_len:
                l = 0
                split_points.append(j)

        start = 0
        for point in split_points:
            paper_chunks.append(papers[start:point])
            title_chunks.append(titles[start:point])
            start = point

        paper_chunks.append(papers[start:])
        title_chunks.append(titles[start:])

        return paper_chunks, title_chunks

    def process_outlines(self, section_outline, sub_outlines):
        res = ''
        survey_title, survey_sections, survey_section_descriptions = self.extract_title_sections_descriptions(section_outline)
        res += f'# {survey_title}\n\n'
        for i in range(len(survey_sections)):
            section = survey_sections[i]
            res += f'## {i+1} {section}\nDescription: {survey_section_descriptions[i]}\n\n'
            subsections, subsection_descriptions = self.extract_subsections_subdescriptions(sub_outlines[i])
            for j in range(len(subsections)):
                subsection = subsections[j]
                res += f'### {i+1}.{j+1} {subsection}\nDescription: {subsection_descriptions[j]}\n\n'
        return res
