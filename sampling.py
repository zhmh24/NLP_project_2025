import json
import torch  # Ensure torch is imported
from train.model import MinecraftLM

def generate_data(prompt, model_path, max_length=9216, top_k=50, top_p=0.9, temperature=1.0):
    """
    Generate data using the trained model.

    Args:
        prompt (str): The input prompt to the model.
        model_path (str): Path to the trained model.
        max_length (int): Maximum length of the generated sequence.
        top_k (int): Top-k sampling.
        top_p (float): Top-p (nucleus) sampling.
        temperature (float): Sampling temperature.

    Returns:
        dict: Generated data in JSON format.
    """
    # Load the trained model to GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinecraftLM()
    model.lm = model.lm.from_pretrained(model_path).to(device)
    tokenizer = model.tokenizer

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Move input data to the same device as the model
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate data
    outputs = model.lm.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # Explicitly pass attention_mask
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature
    )

    # Decode the generated sequence
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Format as JSON
    data = {"data": generated_text}
    return data

def generate_multiple_data(prompts, model_path, max_length=9216, top_k=50, top_p=0.9, temperature=1.0):
    """
    Generate multiple data samples using the trained model.

    Args:
        prompts (list of str): List of input prompts to the model.
        model_path (str): Path to the trained model.
        max_length (int): Maximum length of the generated sequence.
        top_k (int): Top-k sampling.
        top_p (float): Top-p (nucleus) sampling.
        temperature (float): Sampling temperature.

    Returns:
        list of dict: List of generated data in JSON format.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinecraftLM()
    model.lm = model.lm.from_pretrained(model_path).to(device)
    tokenizer = model.tokenizer

    results = []
    for prompt in prompts:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Move input data to the same device as the model
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate data
        outputs = model.lm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Explicitly pass attention_mask
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )

        # Decode the generated sequence
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Append the result
        results.append({"prompt": prompt, "data": generated_text})

    return results

if __name__ == "__main__":
    # # Define the input prompts and model path
    # prompts = ["e e e e e", "f f f f f", "g g g g g", "h h h h h", "i i i i i", "j j j j j", "k k k k k", "l l l l l", "m m m m m", "n n n n n"]
    # model_path = "Qwen-MinecraftLM/iteration_10000"

    # # Generate multiple data samples
    # generated_data = generate_multiple_data(prompts, model_path)

    # # Save the generated data to a JSON file
    # with open("generated_multiple_data.json", "w", encoding="utf-8") as f:
    #     json.dump(generated_data, f, ensure_ascii=False, indent=4)

    # print("Generated multiple data saved to generated_multiple_data.json")

    # Define the input prompt, base input, and model path
    model = MinecraftLM(context_length=9216)
    model_path = "Qwen-MinecraftLM/iteration_10000"
    state_dict = torch.load("Qwen-MinecraftLM/iteration_10000/pytorch_model.bin", map_location="cpu")
    for k, v in state_dict.items():
        print(f"{k}: {v.shape}")
    
    main_lm_state = {k[3:]: v for k, v in state_dict.items() if k.startswith("lm.")}
    
    biome_state = {"weight": state_dict["biome_embedding.weight"]}
    elevation_state = {"weight": state_dict["elevation_embedding.weight"]}
    tree_state = {"weight": state_dict["tree_embedding.weight"]}


    model.lm.load_state_dict(main_lm_state)
    model.biome_embedding.load_state_dict(biome_state)
    model.elevation_embedding.load_state_dict(elevation_state)
    model.tree_embedding.load_state_dict(tree_state)
    prompt = "1 1 1 1 1"

    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    generated_data = model.sample(
        prompt=prompt,
        biome_str="universal_minecraft:desert",
        tree_str="many trees",
        slope_str="low elevation difference",
        max_new_tokens=9216,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
    )

    

    with open("generated_data.json", "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)
    
    print("Generated data saved to generated_data.json")