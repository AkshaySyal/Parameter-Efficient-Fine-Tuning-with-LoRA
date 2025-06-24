# Parameter-Efficient-Fine-Tuning-with-LoRA
demonstrates fine-tuning the Flan-T5 model for dialogue summarization using both full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) with LoRA, evaluating the performance improvements with ROUGE metrics.

# Problem Statement
1. Setup and Initial Model Load
Dependencies and Libraries

Install required packages: datasets, evaluate, rouge_score, peft.

Import essential libraries: torch, time, pandas, numpy.

Hugging Face Transformers

Import core components: AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer.

Dataset Loading

Load the knkarthick/dialogsum dataset from Hugging Face.

Flan-T5 Model Loading

Load google/flan-t5-base model and tokenizer.

Use torch_dtype=torch.bfloat16 for reduced memory usage.

2. Zero-Shot Inferencing (Baseline)
Initial Test

Run inference on sample dialogue (index 42) using the original Flan-T5 model.

Poor Performance

The model often just repeats the first sentence.

Example:

Model Output: "Person1: I'm not sure how to adjust my life."

Human Summary: "Person1 wants to adjust #Person1#'s life and #Person2# suggests #Person1# be positive and stay healthy."

3. Full Fine-Tuning
3.1 Preprocessing the Dataset
Instructional Format

Convert dialogue-summary pairs into instruction format.

Prompt Structure

"Summarize the following conversation.\n" + dialogue + "\nSummary:"

Tokenization

Tokenize dialogue (input) with max_length=1024.

Tokenize summary (label) with max_length=128.

3.2 Fine-Tuning the Model
Trainer API

Use Hugging Face Trainer API.

Training Arguments

batch_size=16, num_train_epochs=1, learning_rate=1e-3

gradient_accumulation_steps=16, eval_steps=10, logging_steps=10

Data Collator

Use DataCollatorForSeq2Seq.

Training Setup

1 epoch, 12,460 examples

Total train batch size: 256, Optimization steps: 48

Trainable parameters: 247,577,856

Model Saving

Save model to ./flan-t5-base-dialogsum-checkpoint.

Model Loading

Load fine-tuned model with AutoModelForSeq2SeqLM.

3.3 Qualitative Evaluation (Human Evaluation)
Improved Summary

Example (index 42):

INSTRUCT MODEL Output: "#Person1# can't sleep well every night and drinks a lot of wine. #Person2# advises #Person1# to do some exercise every morning."

3.4 Quantitative Evaluation (ROUGE Metric)
Metric Used: ROUGE (rouge1, rouge2, rougeL, rougeLsum)

Sample Evaluation

Run on first 10 test examples.

Results

rouge1: +8.00%

rougeL: +2.37%

rougeLsum: +2.43%

rouge2: −2.23%

4. Parameter Efficient Fine-Tuning (PEFT) with LoRA
Definition

LoRA introduces small adapter layers to adapt the LLM with fewer parameters.

Enables efficient reuse of the same base model for multiple tasks.

4.1 Setup the LoRA Model
LoRA Config

task_type=SEQ_2_SEQ_LM, r=32, lora_alpha=32, lora_dropout=0.1

Adapter Layers

Added using get_peft_model().

Trainable Parameters

3,538,944, only 1.41% of original 251M parameters.

4.2 Train the LoRA Adapter
Training Arguments

Similar to full fine-tuning, but possibly higher learning rate.

Training Time

~6 minutes on Google Colab GPU.

Model Saving

Save to ./flan-t5-base-dialogsum-lora.

Model Loading

Load base model and adapter using AutoModelForSeq2SeqLM.

4.3 Qualitative Evaluation (Human Evaluation)
PEFT Summary Quality

Example (index 42):

"#Person1# is pale and doesn't know how to adjust his life. #Person2# advises #Person1# to get plenty of sleep and exercise."

4.4 Quantitative Evaluation (ROUGE Metric)
Sample Generation

Evaluate original, fine-tuned, and PEFT models.

PEFT Performance vs Original

rouge1: +16.78%

rouge2: +3.20%

rougeL: +9.65%

rougeLsum: +9.60%

PEFT vs Full Fine-Tuning

Slight drop in some metrics:

rouge1: −8.79%

rouge2: +5.43%

Trade-Off

PEFT is significantly more efficient with minimal performance drop.
