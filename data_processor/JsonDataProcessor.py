from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

from data_processor.DataProcessor import DataProcessor


class JsonDataProcessor(DataProcessor):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def get_data(self) -> DatasetDict:
        if "model_context_window" in self.config:
            context_window = self.config["model_context_window"]
        else:
            context_window = self.tokenizer.model_max_length

        data = load_dataset("json", data_files=[self.config["data"]["dataset"]])
        data = data.map(lambda data_point: self.tokenizer(
            self._generate_prompt(
                data_point["text"],
                self.tokenizer.eos_token),
            max_length=context_window,
            truncation=True,
        ))
        return data

    def _generate_prompt(self, text, eos_token: str) -> str:
        text += eos_token
        return text
