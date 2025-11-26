# hf https://huggingface.co/docs/transformers/main_classes/text_generation
from starvector.validation.svg_validator_base import SVGValidator, register_validator
import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from starvector.data.util import rasterize_svg
import os

class SVGValDataset(Dataset):
    def __init__(self, dataset_name, config_name, split, im_size, num_samples, processor):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.im_size = im_size
        self.num_samples = num_samples
        self.processor = processor

        # 如果 dataset_name 是本地用 save_to_disk() 保存的数据集目录，则使用 load_from_disk 加载
        # 否则按原逻辑走 datasets.load_dataset（支持 HF Hub / 本地 snapshot 等）
        if os.path.isdir(self.dataset_name) and os.path.isfile(
            os.path.join(self.dataset_name, "dataset_dict.json")
        ):
            from datasets import load_from_disk

            dataset_dict = load_from_disk(self.dataset_name)
            if self.split not in dataset_dict:
                raise ValueError(
                    f"本地数据集中不包含 split='{self.split}'，可用的 split 有：{list(dataset_dict.keys())}"
                )
            self.data = dataset_dict[self.split]
        else:
            if self.config_name:
                self.data = load_dataset(self.dataset_name, self.config_name, split=self.split)
            else:
                self.data = load_dataset(self.dataset_name, split=self.split)
        
        if self.num_samples != -1:
            self.data = self.data.select(range(self.num_samples))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        svg_str = self.data[idx]['Svg']
        sample_id = self.data[idx]['Filename']
        image = rasterize_svg(svg_str, resolution=self.im_size)
        image = self.processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        caption = self.data[idx].get('Caption', "")
        return {
            'Svg': svg_str,
            'image': image,
            'Filename': sample_id,
            'Caption': caption
        }
    
                
@register_validator
class StarVectorHFSVGValidator(SVGValidator):
    def __init__(self, config):
        super().__init__(config)
        # Initialize HuggingFace model and tokenizer here
        self.torch_dtype = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32
        }[config.model.torch_dtype]
        
        # 默认使用基础 HuggingFace 权重（config.model.name）
        base_model_path = config.model.name

        # 如果提供了 from_checkpoint，则在基础模型上加载额外权重
        if config.model.from_checkpoint:
            # 这里的 checkpoint 目录通常是 DeepSpeed / Trainer 输出，不包含 HF config.json
            # 因此我们始终从 base_model_path 读取配置，只从 checkpoint 目录读取权重。
            checkpoint_dir = self.resume_from_checkpoint
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
            ).to(config.run.device)

            # 常见的 DeepSpeed 权重文件位置
            candidate_paths = [
                os.path.join(checkpoint_dir, "pytorch_model", "mp_rank_00_model_states.pt"),
                os.path.join(checkpoint_dir, "pytorch_model.bin"),
                os.path.join(checkpoint_dir, "pytorch_model_fp32.bin"),
            ]

            state_dict_loaded = False
            for ckpt_path in candidate_paths:
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    # 兼容不同的保存格式
                    if isinstance(ckpt, dict):
                        if "module" in ckpt:
                            state_dict = ckpt["module"]
                        elif "model" in ckpt:
                            state_dict = ckpt["model"]
                        elif "state_dict" in ckpt:
                            state_dict = ckpt["state_dict"]
                        else:
                            state_dict = ckpt
                    else:
                        state_dict = ckpt

                    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded checkpoint from {ckpt_path}")
                    if missing:
                        print(f"Missing keys when loading checkpoint: {len(missing)}")
                    if unexpected:
                        print(f"Unexpected keys when loading checkpoint: {len(unexpected)}")
                    state_dict_loaded = True
                    break

            if not state_dict_loaded:
                print(
                    f"Warning: config.model.from_checkpoint={config.model.from_checkpoint} "
                    f"已设置，但在目录 {checkpoint_dir} 下没有找到可用的权重文件，"
                    f"将仅使用基础模型 {base_model_path}。"
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
            ).to(config.run.device)

        # 处理器始终从基础模型目录加载（那里有 processor / tokenizer 配置）
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        
        self.tokenizer = self.model.model.svg_transformer.tokenizer
        self.svg_end_token_id = self.tokenizer.encode("</svg>")[0] 

    def get_dataloader(self):
        self.dataset = SVGValDataset(
            self.config.dataset.dataset_name,
            self.config.dataset.config_name,
            self.config.dataset.split,
            self.config.dataset.im_size,
            self.config.dataset.num_samples,
            self.processor
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers
        )
        return self.dataloader
    
    def release_memory(self):
        # Clear references to free GPU memory
        self.model.model.svg_transformer.tokenizer = None
        self.model.model.svg_transformer.model = None
        
        # Force CUDA garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
    def generate_svg(self, batch, generate_config):
        if generate_config['temperature'] == 0:
            generate_config['temperature'] = 1.0
            generate_config['do_sample'] = False
        outputs = []
        batch['image'] = batch['image'].to('cuda').to(self.torch_dtype)
        # for i, batch in enumerate(batch['svg']):
        if self.task == 'im2svg':
            outputs = self.model.model.generate_im2svg(batch = batch, **generate_config)
        elif self.task == 'text2svg':
            outputs = self.model.model.generate_text2svg(batch = batch, **generate_config)
        return outputs
        