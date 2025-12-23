from train.model import MinecraftLM
from train.dataset import MinecraftChunkDataset
from train.trainer import MinecraftTrainer, TrainingConfig

# 初始化模型和数据集
model = MinecraftLM(context_length=9216)
dataset = MinecraftChunkDataset("data/dataset_train.jsonl", tokenizer=model.tokenizer, context_length=model.context_length)
eval_dataset = MinecraftChunkDataset("data/dataset_test.jsonl", tokenizer=model.tokenizer, context_length=model.context_length)

# 可选：自定义训练参数
config = TrainingConfig(
    output_dir="Qwen-MinecraftLM",
    learning_rate=5e-5,
    batch_size=1,
    total_steps=1000,
    # 其他参数可按需调整
)

# 创建Trainer并开始训练
trainer = MinecraftTrainer(model, dataset, eval_dataset, config)
trainer.train()