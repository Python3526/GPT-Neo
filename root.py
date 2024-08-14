import torch
from settings import max_new_tokens, temperature, top_k, top_p, do_sample, repetition_penalty
from transformers import BloomTokenizerFast, BloomForCausalLM

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


def main(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens[0],
        temperature=temperature[0],
        top_k=top_k[0],
        top_p=top_p[0],
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id)

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if decoded_output.startswith(prompt):
        decoded_output = decoded_output[len(prompt):].strip()

    print(decoded_output)


if __name__ == '__main__':
    while True:
        input_text = str(input("Enter your prompt: "))
        main(input_text)
