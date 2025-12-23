import json
import torch  # Ensure torch is imported
from train.model import MinecraftLM

# def generate_data(prompt, model_path, max_length=9216, top_k=50, top_p=0.9, temperature=1.0):
#     """
#     Generate data using the trained model.

#     Args:
#         prompt (str): The input prompt to the model.
#         model_path (str): Path to the trained model.
#         max_length (int): Maximum length of the generated sequence.
#         top_k (int): Top-k sampling.
#         top_p (float): Top-p (nucleus) sampling.
#         temperature (float): Sampling temperature.

#     Returns:
#         dict: Generated data in JSON format.
#     """
#     # Load the trained model to GPU or CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MinecraftLM()
#     model.lm = model.lm.from_pretrained(model_path).to(device)
#     tokenizer = model.tokenizer

#     # Tokenize the input prompt
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

#     # Move input data to the same device as the model
#     input_ids = inputs["input_ids"].to(device)
#     attention_mask = inputs["attention_mask"].to(device)

#     # Generate data
#     outputs = model.lm.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,  # Explicitly pass attention_mask
#         max_length=max_length,
#         do_sample=True,
#         top_k=top_k,
#         top_p=top_p,
#         temperature=temperature
#     )

#     # Decode the generated sequence
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Format as JSON
#     data = {"data": generated_text}
#     return data

# def generate_multiple_data(prompts, model_path, max_length=9216, top_k=50, top_p=0.9, temperature=1.0):
#     """
#     Generate multiple data samples using the trained model.

#     Args:
#         prompts (list of str): List of input prompts to the model.
#         model_path (str): Path to the trained model.
#         max_length (int): Maximum length of the generated sequence.
#         top_k (int): Top-k sampling.
#         top_p (float): Top-p (nucleus) sampling.
#         temperature (float): Sampling temperature.

#     Returns:
#         list of dict: List of generated data in JSON format.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MinecraftLM()
#     model.lm = model.lm.from_pretrained(model_path).to(device)
#     tokenizer = model.tokenizer

#     results = []
#     for prompt in prompts:
#         # Tokenize the input prompt
#         inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

#         # Move input data to the same device as the model
#         input_ids = inputs["input_ids"].to(device)
#         attention_mask = inputs["attention_mask"].to(device)

#         # Generate data
#         outputs = model.lm.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,  # Explicitly pass attention_mask
#             max_length=max_length,
#             do_sample=True,
#             top_k=top_k,
#             top_p=top_p,
#             temperature=temperature
#         )

#         # Decode the generated sequence
#         generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Append the result
#         results.append({"prompt": prompt, "data": generated_text})

#     return results

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
    prompt = "Here I will provide a string, each letter represents a kind of block in Minecraft, and you need to complete a chunk. [Important]: the requirement is low elevation difference, universal_minecraft:frozen_river biome, little trees. The string is: d d o o o o o o o o o o d d d d #d d o o o o o o o o o o o d d d #d d d o o o o o o o o o o o d d #d d d o o o o o o o o o o o p p #p p p p p p o o o o p p p p p p #p p p p p p p p o p p p p p p p #p p p p p p p p p p p p p p u u #u u u u u u p p p p u u u u 3 a #3 3 X 3 X 3 u u p p a a 3 3 a a #a a a a a a 3 a u M a a a a a a #a a a a a a a a F M F a a a a a #a a a a a a a F F M F F a a a a #a a a a a a a a F M F a a a a a #a a a a a a a F F M F F a a a a #a a a a a a a 3 F M F 3 a a a F #a a a a a a a a a M a a a a a F #$d o o o o o o o o o o d d d d d #d d o o o o o o o o o o d d d d #d d d o o o o o o o o o o d d d #d d d o o o o o o o o o p p 0 p #p p p p p p p p p p p p p p 0 p #p p p p p p p p p p p p p p 0 p #p p p p p p p p p p p p u u 0 u #u u u u u u u u u u u u 3 3 3 3 #3 3 X 3 X 3 3 a a a a a a a a a #a a a a a a a a a a a a a a a a #a a a a a a a a a F a a a a a a #a a a a a a a F F F F F a a a a #a a a a a a a a a F a a a a a a #a a a a a a a F F F F F a a a a #a a a a a a a 3 3 F 3 3 a a a a #a a a a a a a a a a a a a a a a #$"
    model = MinecraftLM(lm_path="Qwen-MinecraftLM/iteration_8000")

    model.to("cuda")
    model.eval()

    generated_data = model.sample(
        prompt=prompt,
        biome_str="universal_minecraft:frozen_river",
        tree_str="little trees",
        slope_str="low elevation difference",
        max_new_tokens=9216,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        
    )

    

    with open("generated_data.json", "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)
    
    print("Generated data saved to generated_data.json")