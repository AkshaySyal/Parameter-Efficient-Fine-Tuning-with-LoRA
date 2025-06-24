# Parameter-Efficient-Fine-Tuning-with-LoRA
demonstrates fine-tuning the Flan-T5 model for dialogue summarization using both full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) with LoRA, evaluating the performance improvements with ROUGE metrics.

# Problem Statement
## 1. Setup and Initial Model Load

- **Dependencies and Libraries**
  - Install required packages: `datasets`, `evaluate`, `rouge_score`, `peft`.
  - Import essential libraries: `torch`, `time`, `pandas`, `numpy`.

- **Hugging Face Transformers**
  - Import components: `AutoModelForSeq2SeqLM`, `AutoTokenizer`, `Seq2SeqTrainingArguments`, `DataCollatorForSeq2Seq`, `Seq2SeqTrainer`.

- **Dataset Loading**
  - Load the `knkarthick/dialogsum` dataset.

- **Flan-T5 Model Loading**
  - Load `google/flan-t5-base` model and tokenizer.
  - Use `torch_dtype=torch.bfloat16` to reduce GPU memory usage.

## 2. Zero-Shot Inferencing (Baseline)

- **Initial Test**
  - Run inference on dialogue (index 42) using original Flan-T5 model.

- **Poor Performance**
  - Model struggles with summarization, often repeating the first sentence.
  - Example:
    - Model Output: "Person1: I'm not sure how to adjust my life."
    - Human Summary: "Person1 wants to adjust #Person1#'s life and #Person2# suggests #Person1# be positive and stay healthy."

## 3. Full Fine-Tuning

- **3.1 Preprocessing the Dataset**
  - Convert dialogue-summary pairs into instruction format.
  - Prompt structure: `"Summarize the following conversation.\n"` + dialogue + `"\nSummary:"`
  - Tokenize inputs with `max_length=1024`, summaries with `max_length=128`.

- **3.2 Fine-Tuning the Model**
  - Use Hugging Face `Trainer` API.
  - Training args: `batch_size=16`, `epochs=1`, `learning_rate=1e-3`, `gradient_accumulation_steps=16`, `eval_steps=10`.
  - Use `DataCollatorForSeq2Seq`.
  - Train model on 12,460 examples for 1 epoch (~10 mins on Colab GPU).
  - Total train batch size: 256. Optimization steps: 48.
  - Trainable parameters: 247,577,856.
  - Save model to `./flan-t5-base-dialogsum-checkpoint`.
  - Load model with `AutoModelForSeq2SeqLM`.

- **3.3 Qualitative Evaluation (Human Evaluation)**
  - Compare summaries for index 42 using original and fine-tuned models.
  - Fine-tuned output: "#Person1# can't sleep well every night and drinks a lot of wine. #Person2# advises #Person1# to do some exercise every morning."

- **3.4 Quantitative Evaluation (ROUGE Metric)**
  - Use ROUGE (rouge1, rouge2, rougeL, rougeLsum).
  - Generate summaries for first 10 test samples.
  - Fine-tuned model improvement:
    - rouge1: +8.00%
    - rouge2: -2.23%
    - rougeL: +2.37%
    - rougeLsum: +2.43%

## 4. Parameter Efficient Fine-Tuning (PEFT) with LoRA

- **Definition**
  - LoRA introduces small trainable adapters to LLMs.
  - Adapter is a few MBs, reused with base LLM at inference.

- **4.1 Setup the LoRA Model**
  - Define `LoraConfig`: `task_type=SEQ_2_SEQ_LM`, `r=32`, `lora_alpha=32`, `lora_dropout=0.1`.
  - Add adapter layers using `get_peft_model`.
  - Trainable parameters: 3,538,944 (1.41% of full model size).

- **4.2 Train the LoRA Adapter**
  - Use `Seq2SeqTrainer` as before.
  - Faster training: ~6 mins on Colab GPU.
  - Train batch size: 256, steps: 48.
  - Save model to `./flan-t5-base-dialogsum-lora`.
  - Load adapter + base model with `AutoModelForSeq2SeqLM`.

- **4.3 Qualitative Evaluation (Human Evaluation)**
  - LoRA output for index 42: "#Person1# is pale and doesn't know how to adjust his life. #Person2# advises #Person1# to get plenty of sleep and exercise."

- **4.4 Quantitative Evaluation (ROUGE Metric)**
  - Evaluate original, fine-tuned, and PEFT models.
  - PEFT model improvement over original:
    - rouge1: +16.78%
    - rouge2: +3.20%
    - rougeL: +9.65%
    - rougeLsum: +9.60%
  - Compared to full fine-tuning:
    - rouge1: -8.79%
    - rouge2: +5.43%
  - Trade-off: Slight performance drop for major gains in efficiency.

