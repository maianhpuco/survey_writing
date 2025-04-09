import ollama

# Name of the model you want to test — make sure it's pulled already
model_name = 'qwq'  # or 'mistral', 'llama3', etc., depending on what you installed

# Create a simple prompt
prompt = "What is the capital of France?"

# Send the prompt to the model
response = ollama.chat(
    model=model_name,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Print the response
print("Model response:\n", response['message']['content'])
