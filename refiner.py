import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm

class SelfCorrectionRefiner:
    def __init__(self, model_name="google/gemma-2-2b-it", device="cuda"):
        """
        Initialize the refiner with Gemma model for critique and refinement
        """
        print(f"Loading {model_name}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate_critique(self, question, solution, max_length=256):
        """
        Generate critique of a solution
        """
        prompt = f"""<start_of_turn>user
Question: {question}

Proposed Solution:
{solution}

Review this solution step by step. Identify any errors in mathematical reasoning, calculations, or logic. Be specific about what went wrong.

Critique:<end_of_turn>
<start_of_turn>model
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        critique = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the critique part (after "model")
        if "<start_of_turn>model" in critique:
            critique = critique.split("<start_of_turn>model")[-1].strip()
        
        return critique
    
    def generate_refinement(self, question, original_solution, critique, max_length=512):
        """
        Generate refined solution based on critique
        """
        prompt = f"""<start_of_turn>user
Question: {question}

Original Solution:
{original_solution}

Critique of the solution:
{critique}

Based on the critique above, provide a corrected step-by-step solution. End your final answer with \\boxed{{answer}}.

Corrected Solution:<end_of_turn>
<start_of_turn>model
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        refinement = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the refinement part
        if "<start_of_turn>model" in refinement:
            refinement = refinement.split("<start_of_turn>model")[-1].strip()
        
        return refinement
    
    def refine_solution(self, question, solution):
        """
        Complete refinement: critique + refinement in one call
        """
        critique = self.generate_critique(question, solution)
        refinement = self.generate_refinement(question, solution, critique)
        
        return {
            'critique': critique,
            'refined_solution': refinement
        }

# Test function
if __name__ == "__main__":
    refiner = SelfCorrectionRefiner()
    
    # Test example
    question = "John has 5 apples. He gives 2 to Mary. How many does he have left?"
    solution = "John starts with 5 apples. He gives away 2. So 5 + 2 = 7. The answer is \\boxed{7}."
    
    result = refiner.refine_solution(question, solution)
    print("Critique:", result['critique'])
    print("\nRefined:", result['refined_solution'])