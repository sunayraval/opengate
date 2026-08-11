import time
import torch
from typing import Dict, List, Any, Optional
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from app.models.base import BaseVisionModel

class HuggingFaceVLM(BaseVisionModel):
    def __init__(self, name: str, model_path: str, device: Optional[str] = None):
        super().__init__(name=name, model_type="vision_language", device=device)
        self.model_path = model_path
        self.processor = None
        self.model = None

    def load_model(self) -> None:
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float16

        # Check if bitsandbytes is installed to prevent VRAM spillover on 12GB cards
        try:
            import bitsandbytes
            from transformers import BitsAndBytesConfig
            # Use 4-bit quantization to drastically reduce memory usage (4GB instead of 8GB)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=target_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print(f"✅ bitsandbytes found! Loading {self.model_name} in 4-bit precision for maximum speed on 12GB GPUs.")
        except ImportError:
            quantization_config = None
            print(f"⚠️ bitsandbytes not found. Loading in full {target_dtype}. Warning: May spill to System RAM and run slow if VRAM is full.")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        except Exception:
            from transformers import AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=target_dtype,
                quantization_config=quantization_config,
                device_map="cuda" if quantization_config else None
            )
        except ValueError:
            try:
                from transformers import AutoModelForImageTextToText
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=target_dtype,
                    quantization_config=quantization_config,
                    device_map="cuda" if quantization_config else None
                )
            except (ValueError, ImportError):
                from transformers import AutoModelForVision2Seq
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=target_dtype,
                    quantization_config=quantization_config,
                    device_map="cuda" if quantization_config else None
                )

        if not quantization_config:
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(device=self.device, dtype=target_dtype)
            elif self.device != "cpu":
                self.model = self.model.to(device=self.device)
            
        self.model.eval()
        self.is_loaded = True

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.is_loaded = False

    def predict_action(self, image: Image.Image, candidate_actions: List[str]) -> Dict[str, Any]:
        return {"action": "unknown", "confidence": 0.0, "all_scores": {}, "model_used": self.model_name, "inference_time_ms": 0.0}

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        return {"bounding_boxes": [], "model_used": self.model_name, "inference_time_ms": 0.0}

    def generate_completion(
        self,
        image: Optional[Image.Image] = None,
        prompt: str = "Describe the visual contents of this image.",
        temperature: float = 0.7,
        max_tokens: int = 512,
        msgs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        if not self.is_loaded:
            self.load_model()
            
        if msgs is None:
            if image is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
        else:
            messages = msgs
            # Normalize message schema for HuggingFace chat templates
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for block in msg["content"]:
                        if block.get("type") == "image_url":
                            block["type"] = "image"

        if hasattr(self.model, "chat"):
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            clean_msgs = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text_content = ""
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            text_content += block.get("text", "")
                else:
                    text_content = str(content)
                clean_msgs.append({"role": role, "content": text_content})
                
            import inspect
            chat_sig = inspect.signature(self.model.chat)
            chat_kwargs = {
                "image": image,
                "msgs": clean_msgs,
                "tokenizer": tokenizer,
                "sampling": temperature > 0,
                "temperature": temperature if temperature > 0 else 0.7,
                "max_new_tokens": max_tokens
            }
            if "context" in chat_sig.parameters:
                chat_kwargs["context"] = None
                
            with torch.inference_mode():
                res = self.model.chat(**chat_kwargs)
                    
            res_str = res[0] if isinstance(res, tuple) else res
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
        else:
            try:
                text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except ValueError:
                # Fallback for models without a registered chat_template
                text_prompt = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    text_content = ""
                    if isinstance(content, list):
                        for block in content:
                            if block.get("type") == "text":
                                text_content += block.get("text", "")
                            elif block.get("type") == "image":
                                text_content += "<image>\n"
                    else:
                        text_content = str(content)
                    text_prompt += f"User: {text_content}\n" if role == "user" else f"Assistant: {text_content}\n"
                text_prompt += "Assistant: "
            
            if image is not None:
                inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt")
            else:
                inputs = self.processor(text=[text_prompt], return_tensors="pt")
                
            # Cast inputs correctly
            compute_dtype = torch.bfloat16 if (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) else torch.float16
            inputs = {k: v.to(self.device, dtype=compute_dtype) if torch.is_floating_point(v) else v.to(self.device) for k, v in inputs.items()}

            input_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0
                )
                
            out_tokens = output_ids[0]
            if out_tokens.shape[0] >= input_len and torch.equal(out_tokens[:input_len], inputs["input_ids"][0]):
                generated_ids = out_tokens[input_len:]
            else:
                generated_ids = out_tokens
    
            res_str = self.processor.decode(generated_ids, skip_special_tokens=True)
            prompt_tokens = len(inputs["input_ids"][0])
            completion_tokens = len(generated_ids)
            total_tokens = prompt_tokens + completion_tokens

        elapsed_ms = self.measure_time_ms(start_time)
        
        # Aggressive memory cleanup to prevent OOM/CUDA device-side asserts over time
        import gc
        if 'inputs' in locals():
            del inputs
        if 'output_ids' in locals():
            del output_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {
            "id": f"cmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "text": res_str,
                    "message": {"role": "assistant", "content": res_str},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "inference_time_ms": round(elapsed_ms, 2),
            "raw_response": res_str
        }
