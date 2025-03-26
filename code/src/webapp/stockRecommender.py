from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load TinyLlama model and tokenizer once (to avoid reloading)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

def recommend_stock(data):
    """
    Given an investor's profile, recommend stocks using TinyLlama.
    """
    investor_profile = f"""
    Investor Profile:
    - Net Worth: {data['net_worth']} USD
    - Liquidity: {data['liquidity']} USD
    - Risk Score: {data['risk_score']}
    - Mortgage Debt: {data['mortgage_debt']} USD
    - Region: {data['region']}
    - Preferred Sector: {data['preferred_sector']}

    Suggest the best stocks for this investor with reasoning.
    """

    # Tokenize input
    inputs = tokenizer(investor_profile, return_tensors="pt").to("cuda")

    # Generate recommendation
    with torch.no_grad():
        output = model.generate(**inputs, max_length=200)

    # Decode response
    recommendation = tokenizer.decode(output[0], skip_special_tokens=True)

    return recommendation