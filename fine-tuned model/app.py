import modal

stub = modal.App("llama3-pricer-finetune")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch==2.5.1+cu124",
        "torchvision==0.20.1+cu124",
        "torchaudio==2.5.1+cu124",
        extra_index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "requests==2.32.3",
        "bitsandbytes==0.46.0",
        "transformers==4.48.3",
        "accelerate==1.3.0",
        "datasets==3.2.0",
        "peft==0.14.0",
        "trl==0.14.0",
        "matplotlib",
        "wandb"
    )
)


@stub.function(
    gpu="A100",
    timeout=60 * 60 * 15,  # 4 hours
    secrets=[modal.Secret.from_name("huggingface-wandb-secrets")],
    image=image
)
def train():
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
    import wandb
    from huggingface_hub import login

    # Constants
    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
    PROJECT_NAME = "pricer"
    HF_USER = "llmengin"
    DATASET_NAME = f"{HF_USER}/pricer-data"
    PROJECT_RUN_NAME = "pricer-2025-07-02_03.45.11"
    HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"
    RUN_NAME = PROJECT_RUN_NAME  # For wandb consistency
    MAX_SEQUENCE_LENGTH = 182

    # Hyperparameters
    LORA_R = 32
    LORA_ALPHA = 64
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    LORA_DROPOUT = 0.1
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 1e-4
    LR_SCHEDULER_TYPE = 'cosine'
    WARMUP_RATIO = 0.03
    OPTIMIZER = "paged_adamw_32bit"
    EPOCHS = 1
    STEPS = 50
    SAVE_STEPS = 2000
    WEIGHT_DECAY = 0.001
    LOG_TO_WANDB = True

    # Setup credentials
    login(token=os.environ["HF_TOKEN"])
    os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    os.environ["WANDB_PROJECT"] = PROJECT_NAME
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_WATCH"] = "gradients"
    wandb.login()

    # Load dataset
    dataset = load_dataset(DATASET_NAME)
    train = dataset['train']

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model from checkpoint if exists
    try:
        model = AutoModelForCausalLM.from_pretrained(
            HUB_MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Resuming from checkpoint.")
    except:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Starting from base model.")

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    collator = DataCollatorForCompletionOnlyLM("Price is $", tokenizer=tokenizer)

    lora_parameters = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    train_params = SFTConfig(
        output_dir=PROJECT_RUN_NAME,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        eval_strategy="no",
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        save_steps=SAVE_STEPS,
        save_total_limit=10,
        logging_steps=STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=True,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="wandb",
        run_name=RUN_NAME,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        dataset_text_field="text",
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_NAME,
        hub_private_repo=True,
        max_steps=100000
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train,
        peft_config=lora_parameters,
        args=train_params,
        data_collator=collator
    )

    trainer.train()
    trainer.model.push_to_hub(PROJECT_RUN_NAME, private=True)
    print(f"Saved to the hub: {PROJECT_RUN_NAME}")
    wandb.finish()
