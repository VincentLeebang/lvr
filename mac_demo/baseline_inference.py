
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

print("Loading vision-language model...")

device = "mps" if torch.backends.mps.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "mps" else torch.float32
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")
print("Device:", device)


def predict(image_path, question):

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100
        )

    answer = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    print("\nQUESTION:", question)
    print("\nMODEL OUTPUT:", answer)

    return answer


if __name__ == "__main__":

    image_path = "mac_demo/images/test.jpg"

    question = "Describe the objects visible in this image."

    predict(image_path, question)
