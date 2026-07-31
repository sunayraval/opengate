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
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
        except ValueError:
            from transformers import AutoModelForVision2Seq
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
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

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        if image is not None:
            inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt")
        else:
            inputs = self.processor(text=[text_prompt], return_tensors="pt")
            
        inputs = inputs.to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )
            
        out_tokens = output_ids[0]
        if out_tokens.shape[0] >= input_len and torch.equal(out_tokens[:input_len], inputs.input_ids[0]):
            generated_ids = out_tokens[input_len:]
        else:
            generated_ids = out_tokens

        res_str = self.processor.decode(generated_ids, skip_special_tokens=True)

        elapsed_ms = self.measure_time_ms(start_time)
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
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(inputs.input_ids[0]) + len(generated_ids)
            },
            "inference_time_ms": round(elapsed_ms, 2),
            "raw_response": res_str
        }
