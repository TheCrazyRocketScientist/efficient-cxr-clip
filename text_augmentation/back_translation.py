import argparse
import ast
import os
import random
import sys
import gc
import wandb

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.marian import MarianMTModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TextDataset(Dataset):
    def __init__(self, tokenizer, original_data_path=None, text_data_list=None):
        assert original_data_path or text_data_list
        self.df, self.text_num_list, self.text_data_list = None, None, None
        self.tokenizer = tokenizer

        if original_data_path:
            self.df = pd.read_csv(original_data_path) if original_data_path is not None else None
            self.text_data_list = []
            self.text_num_list = []

            for idx in range(len(self.df)):

                if "text" in self.df:
                    text_list = ast.literal_eval(self.df["text"][idx])

                else:
                    raise NotImplementedError

                self.text_num_list.append(len(text_list))
                self.text_data_list.extend(text_list)

        if text_data_list:
            self.text_data_list = text_data_list

    def __len__(self):
        return len(self.text_data_list)

    def __getitem__(self, index):
        return self.text_data_list[index]

    def collate_fn(self, instances):
        tokens = self.tokenizer(instances, return_tensors="pt", padding=True,max_length=512, truncation=True)
        return tokens
    
    def _clear_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

class BackTranslation:
    def __init__(self, lang="de"):
        self.lang = lang
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.en_lang_tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}")
        self.lang_en_tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang}-en")

        self.en_lang_translator = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}",torch_dtype=torch.bfloat16,use_safetensors=True).to("cpu")
        self.lang_en_translator = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{lang}-en",torch_dtype=torch.bfloat16,use_safetensors=True).to("cpu")

    def _clear_memory(self):
        """Standard cleanup for high-memory environments"""
        torch.cuda.empty_cache()
        gc.collect()

    def do_back_translation(self, original_data_path, out_data_path, batch_size, temperature, **generate_kwargs):
        assert len(temperature) <= 2
        temp1, temp2 = (temperature[0], temperature[0]) if len(temperature) == 1 else temperature

        pandas_text_dataset = TextDataset(self.en_lang_tokenizer, original_data_path=original_data_path)
        dataloader = DataLoader(
            pandas_text_dataset,
            shuffle=False,  
            drop_last=False,
            num_workers=4,
            batch_size=batch_size,
            collate_fn=pandas_text_dataset.collate_fn,
        )

        self.en_lang_translator.to(self.device)
        self.en_lang_translator.eval()

        text_num_list = pandas_text_dataset.text_num_list
        lang_out_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                lang_out = self.en_lang_translator.generate(**batch.to(self.device), temperature=temp1, **generate_kwargs)
                for out in lang_out:
                    lang_out_list.append(self.en_lang_tokenizer.decode(out).replace("<pad>", "").replace("</s>", "").strip())
        
        self.en_lang_translator.to("cpu")
        self._clear_memory()

        with open(outp_data_path.replace("csv", f"{self.lang}.txt"), "w") as fout:
            fout.write("\n".join(lang_out_list))
            fout.write("\n")
        

        self.lang_en_translator.to(self.device)
        self.lang_en_translator.eval()

        lang_text_dataset = TextDataset(self.lang_en_tokenizer, text_data_list=lang_out_list)
        dataloader = DataLoader(
            lang_text_dataset, shuffle=False, drop_last=False, num_workers=4, batch_size=batch_size, collate_fn=lang_text_dataset.collate_fn
        )

        en_out_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                en_out = self.lang_en_translator.generate(**batch.to(self.device), temperature=temp2, **generate_kwargs)
                for out in en_out:
                    en_out_list.append(self.lang_en_tokenizer.decode(out).replace("<pad>", "").replace("</s>", "").strip())


        self.lang_en_translator.to("cpu")
        self._clear_memory()

        text_augment_list = []
        start = 0

        for text_num in text_num_list:
            text_augment_list.append(en_out_list[start : start + text_num])
            start += text_num

        pandas_text_dataset.df["text_augment"] = text_augment_list
        if "text" not in pandas_text_dataset.df:
            pandas_text_dataset.df["text_original"] = pandas_text_dataset.text_data_list
        pandas_text_dataset.df.to_csv(out_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="it")
    parser.add_argument("--temperature", nargs="+", type=float, required=True, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=5)
    args = parser.parse_args()

    run = wandb.init(
        project="mimic_train",
        config=vars(args),
        job_type="back-translation"
    )

    backtranslation = BackTranslation(lang=args.lang)
    set_random_seed(args.seed)
    inp_data_path = rf"text_augmentation/mimic_train_study_wise_final.csv"
    outp_data_path = f"datasets/mimic_train_augmented.csv"
    if not os.path.exists(os.path.dirname(outp_data_path)):
        os.mkdir(os.path.dirname(outp_data_path))
    backtranslation.do_back_translation(
        inp_data_path, outp_data_path, batch_size=args.bsz, num_beams=args.num_beams, temperature=args.temperature, do_sample=True
    )

    wandb.save(outp_data_path)

    run.finish()
