#!/usr/bin/env python3
import os
import sys
from google import genai
from google.genai import types

def run_token_experiment(prompt: str, model_name: str, temperature: float, top_p: float):
    """
    Executes a streaming prompt using explicit temperature and top_p values.
    Demonstrates token sampling mechanics in real-time.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        print("CRITICAL ERROR: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        print("Please run: export GEMINI_API_KEY='your_key'", file=sys.stderr)
        sys.exit(1)

    print("\n" + "="*60)
    print(f"🔬 EXPERIMENT CONFIGURATION:")
    print(f"   Model:        {model_name}")
    print(f"   Temperature:  {temperature}")
    print(f"   Top-P:        {top_p}")
    print("="*60 + "\n")

    client = genai.Client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=300 
    )

    print(f"Prompt: \"{prompt}\"\n")
    print("Streaming Response: ", end="", flush=True)

    try:
        response_stream = client.models.generate_content_stream(
            model=model_name,  # Dynamic model string
            contents=prompt,
            config=config
        )

        for chunk in response_stream:
            print(chunk.text, end="", flush=True)
        print("\n\n[Inference Stream Complete]")

    except Exception as e:
        print(f"\nAPI Error encountered: {e}", file=sys.stderr)

if __name__ == "__main__":
    # RESTRUCTURED PROMPT: Using clean demarcation to separate instructions from input text
    test_prompt = """
    You are a sci-fi novelist. Complete the following partial sentence with an unpredictable, highly creative sci-fi twist.
    
    CRITICAL: Do NOT repeat the prompt sentence. Start your response immediately with the words that complete the sentence.
    
    Target Sentence:
    'The colony ship's engines unexpectedly stopped because...'
    """
    
    # Corrected Experiment B: Creative/Unpredictable Space
    run_token_experiment(
        prompt=test_prompt, 
        model_name='gemini-2.5-flash', 
        temperature=1.0,  # Flatten the curves so unique words have a higher chance
        top_p=0.95        # Open the door to 95% of the possible vocabulary pool
    )
