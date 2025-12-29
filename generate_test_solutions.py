import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm

class GemmaSolutionGenerator:
    def __init__(self, model_name="google/gemma-2-2b-it", device="cuda"):
        print(f"Loading {model_name}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        print("Model loaded!")
    
    def generate_solution(self, question, temperature=0.7, max_new_tokens=512):
        """Generate a single solution"""
        prompt = f"""Solve this math problem step by step. Show your work and end with your final answer in \\boxed{{answer}} format.

Question: {question}

Solution:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only new tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        solution = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return solution.strip()
    
    def generate_multiple_solutions(self, question, n=5, temperature=0.7):
        """Generate n solutions for a question"""
        solutions = []
        for i in range(n):
            solution = self.generate_solution(question, temperature=temperature)
            solutions.append(solution)
        return solutions

def generate_test_solutions(
    input_file="gsm8k_test_questions.jsonl",
    output_file="gsm8k_test_generations.jsonl",
    n_solutions=5,
    temperature=0.7
):
    """Generate solutions for all test questions"""
    
    # Load questions
    print(f"Loading questions from {input_file}...")
    questions = []
    with open(input_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    
    print(f"Loaded {len(questions)} questions")
    
    # Initialize generator
    generator = GemmaSolutionGenerator()
    
    # Generate solutions
    print(f"\nGenerating {n_solutions} solutions per question...")
    results = []
    
    for item in tqdm(questions, desc="Generating"):
        question = item['question']
        
        # Generate multiple solutions
        solutions = generator.generate_multiple_solutions(
            question, 
            n=n_solutions,
            temperature=temperature
        )
        
        # Save result
        result = {
            'question': question,
            'answer': item['answer'],
            'generated_answers': solutions
        }
        results.append(result)
    
    # Save all results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"✓ Generated solutions for {len(results)} questions")
    print(f"✓ Saved to {output_file}")

if __name__ == "__main__":
    generate_test_solutions(
        input_file="gsm8k_test_questions.jsonl",
        output_file="gsm8k_test_generations.jsonl",
        n_solutions=5,
        temperature=0.7
    )