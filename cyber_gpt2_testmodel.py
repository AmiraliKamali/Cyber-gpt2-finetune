from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 📂 Step 1: Load the Fine-Tuned Model and Tokenizer

model_path = 'cyber_model'  # Path to the fine-tuned model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# 📂 Step 2: Optimized Function for Generating Dynamic and Complete Responses

def generate_dynamic_response(input_text, initial_length=50, max_attempts=10, stop_token="[END]"):
    """Generate a complete and dynamic response from the conversational model."""
    
    input_text = f"User: {input_text} Assistant:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    max_length = initial_length  # Start with a small initial length
    attempt = 0
    response = ""

    while attempt < max_attempts:
        attempt += 1
        print(f"🌀 Generating with max_length={max_length} (Attempt {attempt}/{max_attempts})")
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                min_length=max_length - 20,
                pad_token_id=tokenizer.eos_token_id,
                top_k=100,
                top_p=0.85,
                temperature=1.0,
                do_sample=True,
                early_stopping=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the response part of the Assistant and check for the [END] token
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        if stop_token in response:
            response = response.split(stop_token)[0].strip()
            print("✅ Complete response detected with [END] token.")
            break
        
        # Dynamically increase max_length if the response is not complete
        max_length += 100
        if max_length > 1000:
            print("❌ Maximum length exceeded. Stopping generation.")
            break
    
    # If [END] is not reached after the maximum attempts, do not truncate the text completely
    if stop_token not in response and attempt >= max_attempts:
        print("⚠️ Reached maximum attempts without [END] token.")
    
    return response

# 📂 Step 3: Test the Model with Different Inputs

print("🤖 The model is ready. Ask your question or type 'exit' to quit.")

while True:
    input_text = input("👤 You: ")
    if input_text.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    
    response = generate_dynamic_response(input_text, initial_length=50, max_attempts=15, stop_token="[END]")
    print(f"🤖 Assistant: {response}\n")
