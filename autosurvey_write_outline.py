import os
import json
import argparse
from src.agents.outline_writer import outlineWriter  # Agent to generate the outline
from src.agents.writer import subsectionWriter       # Agent to write subsections
from src.agents.judge import Judge                   # (Unused here) could be used for evaluating/refining
from src.database import database                    # Handles retrieval and embedding-based search
from tqdm import tqdm                                # Progress bar (not used in this script but imported)
import time                                          # Time utility (not used here directly)
import yaml 
import ollama 

# Removes lines that begin with "Description" from the outline text
def remove_descriptions(text):
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not line.strip().startswith("Description")]
    result = '\n'.join(filtered_lines)
    return result

# Generates an outline using the outlineWriter agent
# def write_outline(topic, model, section_num, outline_reference_num, db, api_key, api_url):
    
#     outline_writer = outlineWriter(
#         # model=model,
#         ollama_model=model, 
#         # # api_key=api_key, 
#         # api_url = api_url, 
#         database=db)
    
#     # print(outline_writer.api_model.chat('hello'))  # Debug/test line to confirm LLM is responsive
    
#     outline = outline_writer.draft_outline(
#         topic, outline_reference_num, 30000, section_num)  # Generate outline with references
    
#     return outline, remove_descriptions(outline)  # Return outline with and without descriptions


# Generates an outline using the outlineWriter agent (Ollama-only version)
def write_outline(topic, model, section_num, outline_reference_num, db, api_key=None, api_url=None):
    # Initialize the outline writer with local Ollama model
    outline_writer = outlineWriter(
        ollama_model=model,
        database=db
    )
    #     # Send the prompt to the model
    # response = ollama.chat(
    #     model=model,
    #     messages=[
    #         {"role": "user", "content": 'What is the capital of Vietnam?'}
    #     ]
    # ) 
    # print(response['message']['content'])  # Debug line to confirm LLM is responsive 
    
    # Generate outline with references
    outline = outline_writer.draft_outline(
        topic=topic,
        reference_num=outline_reference_num,
        chunk_size=30000,
        section_num=section_num
    )

    return outline, remove_descriptions(outline)  # Return outline with and without descriptions
 

# Main pipeline logic
def main(args):

    db = database(
        db_path = args.db_path, 
        embedding_model = args.embedding_model
        )  # Initialize the retrieval DB
    
    api_key = args.api_key

    if not os.path.exists(args.saving_path):  # Create output directory if not exist
        os.mkdir(args.saving_path)

    # Generate the outline
    (
        outline_with_description, 
        outline_wo_description
        ) = write_outline(
            args.topic, 
            args.ollama_model, 
            args.section_num, 
            args.outline_reference_num, 
            db, 
            args.api_key, 
            args.api_url
            )
    print("Outline with description:", outline_with_description)  # Debug line to confirm outline generation 
    print("Outline without description:", outline_wo_description)  # Debug line to confirm outline generation 

    # # Write survey subsections (with or without refinement)
    # (
    #     raw_survey, 
    #     raw_survey_with_references,
    #     raw_references, 
    #     refined_survey, 
    #     refined_survey_with_references, 
    #     refined_references
    #     ) = write_subsection(
    #         args.topic, 
    #         args.model, 
    #         outline_with_description, 
    #         args.subsection_len, 
    #         args.rag_num, 
    #         db, 
    #         args.api_key, 
    #         args.api_url)

    # # Save markdown version of the survey
    # with open(f'{args.saving_path}/{args.topic}.md', 'a+') as f:
    #     f.write(refined_survey_with_references)

    # # Save structured JSON version of survey + references
    # with open(f'{args.saving_path}/{args.topic}.json', 'a+') as f:
    #     save_dic = {}
    #     save_dic['survey'] = refined_survey_with_references
    #     save_dic['reference'] = refined_references
    #     f.write(json.dumps(save_dic, indent=4))

# Entry point
# Parses CLI arguments for configuration
def paras_args():
    parser = argparse.ArgumentParser(description='')  # Argument parser for CLI
    parser.add_argument("--db_config", type=str, default="configs/faiss.yaml", help="Path to YAML config file") 
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output/', type=str, help='Directory to save the output survey')
    # parser.add_argument('--model',default='gpt-4o-2024-05-13', type=str, help='Model to use')
    parser.add_argument('--ollama_model',default='qwq', type=str, help='Model to use')  
    parser.add_argument('--topic',default='', type=str, help='Topic to generate survey for')
    parser.add_argument('--section_num',default=7, type=int, help='Number of sections in the outline')
    parser.add_argument('--subsection_len',default=700, type=int, help='Length of each subsection')
    parser.add_argument('--outline_reference_num',default=1500, type=int, help='Number of references for outline generation')
    parser.add_argument('--rag_num',default=60, type=int, help='Number of references to use for RAG')
    parser.add_argument('--api_url',default='https://api.openai.com/v1/chat/completions', type=str, help='url for API request')
    parser.add_argument('--api_key',default='', type=str, help='API key for the model')
    # parser.add_argument('--db_path',default='./database', type=str, help='Directory of the database.')
    parser.add_argument('--embedding_model',default='nomic-ai/nomic-embed-text-v1', type=str, help='Embedding model for retrieval.')
    args = parser.parse_args()
    
    with open(args.db_config, 'r') as f:
        db_config = yaml.safe_load(f)  
    args.db_path = db_config['db_path']  # Load database path from config 
    print("Database path:", args.db_path)  # Debug line to confirm DB path

    return args 


if __name__ == '__main__':
    args = paras_args()  # Parse CLI args
    main(args)           # Execute the pipeline
