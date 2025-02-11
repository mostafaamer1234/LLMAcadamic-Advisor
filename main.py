import os
from huggingface_hub import hf_hub_download
HUGGING_FACE_API_KEY = os.environ.get("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")
model_id = "mostafatarek4/FitReccomendation"
filenames = [
        ".gitattributes", "README.md", "adapter_config.json", "adapter_model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "train.csv"
]
for filename in filenames:
        downloaded_model_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    token=HUGGING_FACE_API_KEY
        )
        print(downloaded_model_path)




from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipeline = pipeline("text2text-generation", model=model, device=-1, tokenizer=tokenizer, max_length=1000)

pipeline("I have a jacket and a coat and I don't know which one to choose to look fashionable on school campus today")