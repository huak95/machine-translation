import torch

# Clear Ram
torch.cuda.empty_cache()

# 1. Input
source_lang = 'th'
target_lang = 'en'
model_checkpoint = "Helsinki-NLP/opus-mt-th-en"
model_name = 'mt-align'
metric_name = "sacrebleu"
data_path = "df_1340K_SCB+LST+QED+Tatoeba.csv"
data_name = 'LST'
data_rows = True  # Load All Data
# Training Params
batch_size = 34
num_train_epochs = 10

repo_model_name = f'{model_name}-finetuned-{data_name}-{source_lang}-to-{target_lang}'

print('model_checkpoint: ', model_checkpoint)
print('repo_model_name: ', repo_model_name)
print('metric_name: ', metric_name)
print('data_path: ', data_path)
print('data_rows: ', data_rows)


# 2. Prepare Datasets -----------------------------------------------
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric, DatasetDict, Dataset
import pandas as pd
import numpy as np

metric = load_metric(metric_name)

def pre_process_from_csv(path, n_row=100000):
    if n_row != True:
        df_5M = pd.read_csv(path, nrows=n_row)
    else:
        df_5M = pd.read_csv(path)
    list_5M = df_5M.to_dict('records')
    list_sub = ['LST_Corpus']*len(list_5M)
    dict_5M = pd.DataFrame({"translation": list_5M, "subdataset": list_sub})
    return dict_5M

raw_datasets = pre_process_from_csv(data_path, n_row=data_rows)

cut_datasets = DatasetDict()
cut_datasets = Dataset.from_pandas(raw_datasets, split="train+validation").train_test_split(0.025)
print('cut_datasets: \n', cut_datasets)

# 3. Pre-Processing Data ------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Set Source and Target Language
if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "translate Thai to English: "
elif model_checkpoint in ["Helsinki-NLP/opus-mt-en-mul"]:
    prefix = '>>tha<<'
else:
    prefix = ""

max_input_len  = 128
max_target_len = 128

def preprocess_function(examples):
#     inputs = [prefix + ex[source_lang] for ex in examples['translation']]
    inputs, targets = [], []
    for ex_ in (examples['translation']):
        ex_target = ex_[target_lang]
        # SKIP NULL THING
        if ex_target is not None and ex_[source_lang] is not None:
            targets.append(ex_[target_lang])
            inputs.append(prefix + ex_[source_lang])
            
    model_inputs = tokenizer(inputs, max_length=max_input_len, truncation=True) # Pad to longest word (128 char)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_len, truncation=True)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs
    
tokenized_datasets = cut_datasets.map(preprocess_function, 
                                      batched=True, 
                                      batch_size=1000, 
                                      num_proc=32, 
                                      remove_columns=["translation", "subdataset"], # Prevent ArrowInvalid Error (Skip)
                                     )

print('tokenized_datasets: \n', tokenized_datasets)

# 4. Fine Tuning Model ------------------------------------------

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
torch.cuda.empty_cache()
args = Seq2SeqTrainingArguments(
    repo_model_name,
    evaluation_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size  = batch_size//2,
#     gradient_checkpointing=True, # For Accelerate Training
    weight_decay =0.01,
    save_total_limit = 3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    dataloader_num_workers=32, # Multi-tread CPU
    fp16=True,
    gradient_accumulation_steps=10,
    eval_accumulation_steps = 10, # Reduce Using GPU Ram When Evaulation
    optim="adafactor", # Faster than ADAM
    push_to_hub = True,
    report_to="wandb",
)

# 4.1 Create Data collator ----------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels= [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    print('preds', preds)
    print('labels', labels)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens = True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result['score']}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# 4.2 Create Seq2SeqTrainer ----------------------

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset= tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['test'],
#     eval_dataset = None,
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics,
)


# 5. Trining Time --------------------------------------------------
print('-'*50)
print(""""
   _____________   ___  ______  _________  ___   _____  _______  _______
  / __/_  __/ _ | / _ \/_  __/ /_  __/ _ \/ _ | /  _/ |/ /  _/ |/ / ___/
 _\ \  / / / __ |/ , _/ / /     / / / , _/ __ |_/ //    // //    / (_ / 
/___/ /_/ /_/ |_/_/|_| /_/     /_/ /_/|_/_/ |_/___/_/|_/___/_/|_/\___/  
""")
print('-'*50)

trainer.train()
trainer.push_to_hub()

print('-'*50)
print("""
   _________  ____________ __  _________  ___   _____  _______  _______
  / __/  _/ |/ /  _/ __/ // / /_  __/ _ \/ _ | /  _/ |/ /  _/ |/ / ___/
 / _/_/ //    // /_\ \/ _  /   / / / , _/ __ |_/ //    // //    / (_ / 
/_/ /___/_/|_/___/___/_//_/   /_/ /_/|_/_/ |_/___/_/|_/___/_/|_/\___/  
""")
print('-'*50)
