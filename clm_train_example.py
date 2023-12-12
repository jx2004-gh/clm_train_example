'''
Simplified training code of causal language models based on the following reference: 
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
Can be used to run small models with limited dataset on a regular laptop.
'''

# Import required packages
import os
os.environ['TRANSFORMERS_CACHE'] = './cache/'	# replace with your cache directory path
os.environ['HF_DATASETS_CACHE'] = './cache/'	# replace with your cache directory path

import math
from itertools import chain
import evaluate
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# Extract train, eval loss from trainer_state json file
def extract_train_eval_loss(trainer_state_file):
    # Read the JSON file
    with open(trainer_state_file, 'r') as f:
        data = json.load(f)

    # Extract log_history
    log_history = data['log_history']

    # Extract train data
    df_train_data = [(entry['step'], 
                      entry['epoch'], 
                      entry.get('learning_rate', None), 
                      entry.get('loss', None)) 
                     for entry in log_history if 'learning_rate' in entry and 'loss' in entry]
    df_train = pd.DataFrame(df_train_data, columns=['step', 'epoch', 'learning_rate', 'loss'])

    # Extract eval data
    df_eval_data = [(entry['step'], 
                     entry['epoch'], 
                     entry.get('eval_loss', None), 
                     entry.get('eval_accuracy', None)) 
                    for entry in log_history if 'eval_loss' in entry and 'eval_accuracy' in entry]
    df_eval = pd.DataFrame(df_eval_data, columns=['step', 'epoch', 'eval_loss', 'eval_accuracy'])

    return df_train, df_eval


# Plot training loss and eval loss versus step
def plot_loss_vs_step(df_train, df_eval):
    # Plotting training loss
    plt.plot(df_train['step'], df_train['loss'], '-o', 
             label='Training Loss', color='blue')
    
    # Plotting eval loss
    plt.plot(df_eval['step'], df_eval['eval_loss'], '-o', 
             label='Evaluation Loss', color='red')
    
    # Add labels, title, and legend
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss vs. Step')
    plt.legend()
    plt.show()

    
def main():
    model_name_or_path = 'roneneldan/TinyStories-1M'    # replace with your model name or path
    tokenizer_name = 'EleutherAI/gpt-neo-125M'          # replace with your tokenizer
    use_fast_tokenizer = True
    torch_dtype = None    
    dataset_name = None
    dataset_config_name = None 
    train_file = 'VBT.txt'                              # replace with your train file
    validation_file = 'VBT_val.txt'                     # replace with your validation file
    max_train_samples = 100
    max_eval_samples = 20
    block_size = 512
    validation_split_percentage = 10
    keep_linebreaks = True 
    
    training_args = TrainingArguments(
        output_dir = './output',            # output directory for model predictions and checkpoints
        overwrite_output_dir = True,        # overwrite output directory
        do_train = True,                    # do training
        do_eval =True,                      # do evaluation
        per_device_train_batch_size = 4,    # batch size for training
        per_device_eval_batch_size = 4,     # batch size for evaluation
        num_train_epochs = 1,               # total number of training epochs
        learning_rate = 5e-5,               # initial learning rate for AdamW optimizer
        logging_steps = 10,                 # log & save weights each logging_steps
        save_steps = 100,                   # save the model every save_steps
        eval_steps = 10,                    # evaluate the model every eval_steps
        evaluation_strategy = 'steps',      # evaluation strategy to adopt during training
        push_to_hub = False,                # push model to the Hugging Face model hub
        logging_first_step = True,          # whether to log and evaluate before any training steps
    )
        
    # Set seed
    set_seed(training_args.seed)

    # Get datasets
    if dataset_name is not None:
        # Load dataset by name
        raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
        )
        # Split dataset to train and validation if there is no sepcified validation dataset
        if 'validation' not in raw_datasets.keys():
            raw_datasets['validation'] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f'train[:{validation_split_percentage}%]',
            )
            raw_datasets['train'] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f'train[{validation_split_percentage}%:]',
            )
    else:
        # Load dataset from files
        data_files = {}
        dataset_args = {}
        if train_file is not None:
            data_files['train'] = train_file
        if validation_file is not None:
            data_files['validation'] = validation_file
        extension = (
            train_file.split('.')[-1]
            if train_file is not None
            else validation_file.split('.')[-1]
        )
        if extension == 'txt':
            extension = 'text'
            dataset_args['keep_linebreaks'] = keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            **dataset_args,
        )
        # Split dataset to train and validation if there is no specified validation data file
        if 'validation' not in raw_datasets.keys():
            raw_datasets['validation'] = load_dataset(
                extension,
                data_files=data_files,
                split=f'train[:{validation_split_percentage}%]',
                **dataset_args,
            )
            raw_datasets['train'] = load_dataset(
                extension,
                data_files=data_files,
                split=f'train[{validation_split_percentage}%:]',
                **dataset_args,
            )
    
    # Set tokenizer
    tokenizer_kwargs = {
        'use_fast': use_fast_tokenizer,
    }
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            'You are instantiating a new tokenizer from scratch. This is not supported by this script.'
            'You can do it from another script, save it, and load it from here, using --tokenizer_name.'
        )

    if model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool('.ckpt' in model_name_or_path),
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError(
            'Model name or path is missing.'
        )

    # Resize embeddings if necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocess datasets
    if training_args.do_train:
        column_names = list(raw_datasets['train'].features)
    else:
        column_names = list(raw_datasets['validation'].features)

    text_column_name = 'text' if 'text' in column_names else column_names[0]
    
    #Tokenize input texts
    def tokenize_function(texts):
        return tokenizer(texts[text_column_name])

    with training_args.main_process_first(desc='dataset map tokenization'):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc='Running tokenizer on dataset',
        )
    
    # Set block size: max block size is 1024
    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            block_size = 1024
    else:
        block_size = min(block_size, tokenizer.model_max_length)

    # Concatenate all texts from dataset and generate chunks of block_size
    def process_texts(texts):
        # Concatenate all texts
        concatenated_texts = {k: list(chain(*texts[k])) for k in texts.keys()}
        total_length = len(concatenated_texts[list(texts.keys())[0]])
        
        # Drop the small remainder if the total_length is not a multiple of block_size
        total_length = (total_length // block_size) * block_size
        
        # Split by chunks of max_len
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_texts.items()
        }
        result['labels'] = result['input_ids'].copy()
        return result

    with training_args.main_process_first(desc='grouping texts together'):
        lm_datasets = tokenized_datasets.map(
            process_texts,
            batched=True,
            desc=f'Grouping texts in chunks of {block_size}',
        )
    
    if training_args.do_train:
        if 'train' not in tokenized_datasets:
            raise ValueError('--do_train requires a train dataset')
        train_dataset = lm_datasets['train']
        if max_train_samples is not None:
            max_train_samples = min(len(train_dataset), max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if 'validation' not in tokenized_datasets:
            raise ValueError('--do_eval requires a validation dataset')
        eval_dataset = lm_datasets['validation']
        if max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load('accuracy')

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # Train
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics

        max_train_samples = (
            max_train_samples if max_train_samples is not None else len(train_dataset)
        )
        metrics['train_samples'] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

    # Evaluate
    if training_args.do_eval:
        print('*** Evaluate ***')

        metrics = trainer.evaluate()

        max_eval_samples = max_eval_samples if max_eval_samples is not None else len(eval_dataset)
        metrics['eval_samples'] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics['eval_loss'])
        except OverflowError:
            perplexity = float('inf')
        metrics['perplexity'] = perplexity

        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    kwargs = {'finetuned_from': model_name_or_path, 'tasks': 'text-generation'}
    if dataset_name is not None:
        kwargs['dataset_tags'] = dataset_name
        if dataset_config_name is not None:
            kwargs['dataset_args'] = dataset_config_name
            kwargs['dataset'] = f'{dataset_name} {dataset_config_name}'
        else:
            kwargs['dataset'] = dataset_name
    
    # Plot loss vs. step
    trainer_state_file = training_args.output_dir + '/trainer_state.json'
    df_train, df_eval = extract_train_eval_loss(trainer_state_file)
    plot_loss_vs_step(df_train, df_eval)
    
if __name__ == '__main__':
    main()