# import torch

# path = "/home/guanjian/transformers_model"
# tokenizer = RobertaTokenizer.from_pretrained('%s/roberta-large'%path)
# model = RobertaForSequenceClassification.from_pretrained('%s/roberta-large'%path)

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# logits = outputs.logits


import copy
import logging
import os
import torch
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import json
import numpy as np
from datasets import load_dataset, load_metric
# from transformers import RobertaTokenizer, RobertaForSequenceClassification
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    # BertForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    EvalPrediction,
    HfArgumentParser,
    # PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task"},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    def __post_init__(self):
        if self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            extension = self.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings_labels_dict):
        self.encodings = encodings_labels_dict["encodings"]
        self.labels = encodings_labels_dict["labels"]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

import torch.nn as nn
from transformers import BertPreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput

class MyModel(BertPreTrainedModel):
    def __init__(self, config, model_args, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.sp_id = tokenizer.convert_tokens_to_ids("[P]")
        try:
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,)
        except Exception as e:
            print("*"*10+"train from scratch"+"*"*10)
            self.bert = AutoModelForSequenceClassification.from_config(config=config)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.num_labels = config.num_labels
        self.loss = nn.CrossEntropyLoss()
    def forward(self,
                input_ids=None, # Indices of input sequence tokens in the vocabulary
                attention_mask=None, 
                token_type_ids=None, 
                position_ids=None, # Indices of positions of each input sequence tokens in the position embeddings. Selected in the range [0, config.max_position_embeddings - 1]
                head_mask=None, 
                inputs_embeds=None, 
                labels=None, 
                output_attentions=None, 
                output_hidden_states=None, 
                return_dict=None
    ):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )

        # [batch_size, length, hidden_size]
        encoder_hidden_states = output.hidden_states[-1] #outputs["encoder_last_hidden_state"]
        # [batch_size, length]
        mask1 = torch.eq(input_ids, torch.tensor(self.sp_id).to(input_ids.device)).float()
        mask2 = torch.eq(input_ids, torch.tensor(self.tokenizer.mask_token_id).to(input_ids.device)).float()
        # [batch_size, length]
        logits = torch.sum(torch.matmul(encoder_hidden_states*mask1[:, :, None], torch.transpose(encoder_hidden_states*mask2[:, :, None], 1, 2)), 1)
        logits -= (1 - mask2) * (1e20)

        mask3 = torch.eq(torch.cumsum(torch.eq(torch.cumsum(mask2, 1), labels[:, None]).float(), 1), 1).float()
        loss = -torch.mean(torch.sum(torch.log_softmax(logits, dim=-1)*mask3, 1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions
        )

    def save_pretrained(self,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs):
        self.bert.save_pretrained(save_directory, save_config, state_dict, save_function, push_to_hub, **kwargs)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset(
            "csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file, "test": data_args.test_file}
        )
    else:
        # Loading a dataset from local json files
        # datasets = load_dataset(
        #     "json", data_files={"train": data_args.train_file, "validation": data_args.validation_file, "field": "data"}
        # )
        datasets = {}
        with open(data_args.train_file) as fin:
            datasets["train"] = json.load(fin)['data']
        with open(data_args.validation_file) as fin:
            datasets["validation"] = json.load(fin)['data']
        with open(data_args.test_file) as fin:
            datasets['test'] = json.load(fin)['data']

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    is_regression = False
    label_list = list(range(1, 12))
    label_list.sort()
    num_labels = len(label_list)
    print(label_list)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        # finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    print(tokenizer.convert_tokens_to_ids("[P]"))
    tokenizer.add_special_tokens({"additional_special_tokens": ['[P]']})
    print(model_args)
    logging.info(training_args)
    logging.info('=========reinit model=========')
    model = MyModel(
        config,
        model_args,
        tokenizer=tokenizer,
    )


    num_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name, param.size(), torch.numel(param))
            num_param += torch.numel(param)
    print("="*10)
    print("# Parameters:", num_param)
    vocab_size = len(tokenizer)
    print("vocab_size:", vocab_size)
    print("="*10)

    logging.info('=========model inited=========')

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = data_args.max_seq_length

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i+1 for i, v in enumerate(label_list)}
    print(label_to_id)

    def preprocess_function(examples):
        result = {}
        result["encodings"] = tokenizer([l["text"] for l in examples], truncation=True, padding=padding, max_length=max_length)
        result["labels"] = [label_to_id[l["label"]] for l in examples]
        return result
    def magic_process(examples): # provide len = 200
        result = {}
        new_examples = []
        for l in examples:
            tmp_len = len(tokenizer([l["text"]], truncation=True, padding=padding, max_length=max_length)["input_ids"][0])
            if tmp_len > data_args.max_seq_length:
                continue
            else:
                new_examples.append(l)
        examples = copy.deepcopy(new_examples)
        result["encodings"] = tokenizer([l["text"] for l in examples], truncation=True, padding=padding, max_length=max_length)
        result["labels"] = [label_to_id[l["label"]] for l in examples]
        print(len(result['encodings']['input_ids']))
        tmp = [
            {
                'encodings': {
                    'input_ids': result['encodings']['input_ids'][i : i + 200],
                    'attention_mask': result['encodings']['attention_mask'][i : i + 200]
                },
                'labels': result['labels'][i : i + 200]
            }
            for i in range(0, len(result['encodings']['input_ids']), 200)
        ]
        print(len(tmp))
        return tmp
    logging.info("processing datasets")
    train_dataset = OurDataset(preprocess_function(datasets["train"]))
    test_datasets = []
    eval_datasets = []
    for test_set in magic_process(datasets["test"]):
        test_datasets.append(OurDataset(test_set))
    for eval_set in magic_process(datasets["validation"]):
        eval_datasets.append(OurDataset(eval_set))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    logging.info('init trainer')
    # Initialize our Trainer
    print(training_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets[0],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )

    # Training
    eval_results = {}
    tasks = [data_args.task_name] * len(eval_datasets)

    if training_args.do_train:
        trainer.train(
            None
        )
        trainer.save_model()
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        acc = 0
        loss = 0
        size = 0
        cnt = 0
        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)
            print(eval_result)
            size += len(eval_dataset)
            loss += eval_result['eval_loss'] * len(eval_dataset)
            acc += eval_result['eval_accuracy'] * len(eval_dataset)
            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{cnt}_{task}.txt")
            cnt += 1
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {cnt} {task} *****")
                    for key, value in eval_result.items():
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")
            eval_results.update(eval_result)
        output_eval_file = os.path.join(training_args.output_dir, f"eval_results_total_{task}.txt")
        with open(output_eval_file, "w") as writer:
            loss /= size
            acc /= size
            logger.info(f"***** Eval results total {task} *****")
            logger.info(f"  eval_loss = {loss}")
            logger.info(f"  eval_accuracy = {acc}")
            writer.write(f"eval_loss = {loss}\n")
            writer.write(f"eval_accuracy = {acc}\n")

    if training_args.do_predict:
        logger.info("*** Test ***")
        tasks = [data_args.task_name] * len(test_datasets)
        logging.info(f"  test sets numbers = {len(test_datasets)}")
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])
        res = {
            'test_accuracy': 0,
            'cnt': 0
        }
        for test_dataset, task in zip(test_datasets, tasks):
            predictions = trainer.predict(test_dataset=test_dataset)
            value = compute_metrics(predictions)
            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                logger.info(f"***** test results {task} *****")
                logger.info(f"  Test acc = {value['accuracy']}")
            res['test_accuracy'] += value['accuracy']
            res['cnt'] += 1
            logger.info(f"  Total test_accuracy = {res['test_accuracy'] / res['cnt']}")
    return eval_results

if __name__ == "__main__":
    main()
