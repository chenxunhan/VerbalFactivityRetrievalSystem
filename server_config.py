host = "0.0.0.0"
port = "7369"

base_model_path = './JudgeLLM/chatglm3-6b'
checkpoint_path = './JudgeLLM/checkpoints/checkpoint-6000'
tokenizer_path = './JudgeLLM/chatglm3-6b'

max_input_tokens = 512 # 微调的配置
max_output_tokens = 4 # 768最佳
temperature=0.1

CUDA_devices = '2'

api_key = ""
