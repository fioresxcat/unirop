import os
os.environ['HF_HOME'] = '/data/tungtx2/tmp/huggingface'
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .utils import *


tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")
model_en2vi = torch.compile(model_en2vi)
device_en2vi = torch.device("cuda")
model_en2vi.to(device_en2vi).eval()


def translate_en2vi(en_texts: str) -> str:
    input_ids = tokenizer_en2vi(en_texts, padding=True, return_tensors="pt").to(device_en2vi)
    with torch.no_grad():
        output_ids = model_en2vi.generate(
            **input_ids,
            decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
    vi_texts = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    return vi_texts


def main():
    dir = 'data/ReadingBank/train1'
    out_dir = 'data/ReadingBank/train1_vi'
    os.makedirs(out_dir, exist_ok=True)
    jpaths = list(Path(dir).glob('*.json'))
    max_batch_size = 20
    processed_files = os.listdir(out_dir)
    for index, jp in enumerate(jpaths):
        if jp.name in processed_files:
            print(f'Skipping {jp}')
            continue
        with open(jp, 'r') as f:
            list_segments = json.load(f)
        all_texts = [seg['text'] for seg in list_segments]
        all_translated_texts = []
        index = 0
        while index < len(all_texts):
            batch_texts = all_texts[index:index+max_batch_size]
            batch_translated_texts = translate_en2vi(batch_texts)
            all_translated_texts.extend(batch_translated_texts)
            index += max_batch_size
        assert len(all_translated_texts) == len(all_texts)

        for seg_index, translated_text in enumerate(all_translated_texts):
            list_segments[seg_index]['text'] = translated_text

        with open(os.path.join(out_dir, jp.name), 'w') as f:
            json.dump(list_segments, f, ensure_ascii=False)
        print(f'done {jp}')


if __name__ == '__main__':
    main()