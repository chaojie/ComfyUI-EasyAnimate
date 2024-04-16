import os

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)

import sys
easyanimate_path=f'{comfy_path}/custom_nodes/ComfyUI-EasyAnimate'
sys.path.insert(0,easyanimate_path)

import torch
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from omegaconf import OmegaConf

from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from easyanimate.models.transformer3d import Transformer3DModel
from easyanimate.pipeline.pipeline_easyanimate import EasyAnimatePipeline
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import save_videos_grid
from einops import rearrange

class EasyAnimateLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixart_path": (os.listdir(folder_paths.get_folder_paths("diffusers")[0]), {"default": "PixArt-XL-2-512x512"}),
                "motion_ckpt": (folder_paths.get_filename_list("checkpoints"), {"default": "easyanimate_v1_mm.safetensors"}),
                "sampler_name": (["Euler","Euler A","DPM++","PNDM","DDIM"],{"default":"DPM++"}),
                "device":(["cuda","cpu"],{"default":"cuda"}),
            },
        }

    RETURN_TYPES = ("EasyAnimateModel",)
    FUNCTION = "run"
    CATEGORY = "EasyAnimate"

    def run(self,pixart_path,motion_ckpt,sampler_name,device):
        pixart_path=os.path.join(folder_paths.get_folder_paths("diffusers")[0],pixart_path)
        # Config and model path
        config_path         = f"{easyanimate_path}/config/easyanimate_video_motion_module_v1.yaml"
        model_name = pixart_path
        #model_name          = "models/Diffusion_Transformer/PixArt-XL-2-512x512"

        # Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" and "DDIM"
        sampler_name        = "DPM++"

        # Load pretrained model if need
        transformer_path    = None
        motion_module_path = folder_paths.get_full_path("checkpoints", motion_ckpt)
        #motion_module_path  = "models/Motion_Module/easyanimate_v1_mm.safetensors" 
        vae_path            = None
        lora_path           = None

        weight_dtype        = torch.float16
        guidance_scale      = 6.0
        seed                = 43
        num_inference_steps = 30
        lora_weight         = 0.55

        config = OmegaConf.load(config_path)

        # Get Transformer
        transformer = Transformer3DModel.from_pretrained_2d(
            model_name, 
            subfolder="transformer",
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs'])
        ).to(weight_dtype)

        if transformer_path is not None:
            print(f"From checkpoint: {transformer_path}")
            if transformer_path.endswith("safetensors"):
                from safetensors.torch import load_file, safe_open
                state_dict = load_file(transformer_path)
            else:
                state_dict = torch.load(transformer_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

            m, u = transformer.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        if motion_module_path is not None:
            print(f"From Motion Module: {motion_module_path}")
            if motion_module_path.endswith("safetensors"):
                from safetensors.torch import load_file, safe_open
                state_dict = load_file(motion_module_path)
            else:
                state_dict = torch.load(motion_module_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

            m, u = transformer.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}, {u}")

        # Get Vae
        if OmegaConf.to_container(config['vae_kwargs'])['enable_magvit']:
            Choosen_AutoencoderKL = AutoencoderKLMagvit
        else:
            Choosen_AutoencoderKL = AutoencoderKL
        vae = Choosen_AutoencoderKL.from_pretrained(
            model_name, 
            subfolder="vae", 
            torch_dtype=weight_dtype
        )

        if vae_path is not None:
            print(f"From checkpoint: {vae_path}")
            if vae_path.endswith("safetensors"):
                from safetensors.torch import load_file, safe_open
                state_dict = load_file(vae_path)
            else:
                state_dict = torch.load(vae_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

            m, u = vae.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        # Get Scheduler
        Choosen_Scheduler = scheduler_dict = {
            "Euler": EulerDiscreteScheduler,
            "Euler A": EulerAncestralDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler, 
            "PNDM": PNDMScheduler,
            "DDIM": DDIMScheduler,
        }[sampler_name]
        scheduler = Choosen_Scheduler(**OmegaConf.to_container(config['noise_scheduler_kwargs']))

        pipeline = EasyAnimatePipeline.from_pretrained(
            model_name,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=weight_dtype
        )
        pipeline.to(device)
        pipeline.enable_model_cpu_offload()

        if lora_path is not None:
            pipeline = merge_lora(pipeline, lora_path, lora_weight)
        return (pipeline,)

class EasyAnimateRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("EasyAnimateModel",),
                "prompt":("STRING",{"multiline": True, "default":"A snowy forest landscape with a dirt road running through it. The road is flanked by trees covered in snow, and the ground is also covered in snow. The sun is shining, creating a bright and serene atmosphere. The road appears to be empty, and there are no people or animals visible in the video. The style of the video is a natural landscape shot, with a focus on the beauty of the snowy forest and the peacefulness of the road."}),
                "negative_prompt":("STRING",{"multiline": True, "default":"Strange motion trajectory, a poor composition and deformed video, worst quality, normal quality, low quality, low resolution, duplicate and ugly"}),
                "video_length":("INT",{"default":80}),
                "num_inference_steps":("INT",{"default":30}),
                "width":("INT",{"default":512}),
                "height":("INT",{"default":512}),
                "guidance_scale":("FLOAT",{"default":6.0}),
                "seed":("INT",{"default":1234}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "EasyAnimate"

    def run(self,model,prompt,negative_prompt,video_length,num_inference_steps,width,height,guidance_scale,seed):
        generator = torch.Generator(device='cuda').manual_seed(seed)

        with torch.no_grad():
            videos = model(
                prompt, 
                video_length = video_length,
                negative_prompt = negative_prompt,
                height      = height,
                width       = width,
                generator   = generator,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,
            ).videos

            videos = rearrange(videos, "b c t h w -> b t h w c")

            return videos

NODE_CLASS_MAPPINGS = {
    "EasyAnimateLoader":EasyAnimateLoader,
    "EasyAnimateRun":EasyAnimateRun,
}