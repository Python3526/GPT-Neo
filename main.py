from transformers import BloomTokenizerFast, BloomForCausalLM
import torch

# Load the tokenizer and model
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


def main(prompt):
    # Encode input text
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate output with optimized parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,  # Adjust based on needs
        temperature=0.7,  # Adjust based on needs
        top_k=30,
        top_p=0.85,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and print output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from the output if it appears
    if decoded_output.startswith(prompt):
        decoded_output = decoded_output[len(prompt):].strip()

    print(decoded_output)


if __name__ == '__main__':
    while True:
        input_text = str(input("Enter a clear and specific prompt: "))
        main(input_text)
