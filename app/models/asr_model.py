import logging
import torch
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

logger = logging.getLogger(__name__)

class NeMoASRModel:
    def __init__(self, model_name="nvidia/parakeet-tdt-0.6b-v2"):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        
    def load(self):
        if not NEMO_AVAILABLE:
            raise RuntimeError("NeMo toolkit is not installed. Cannot load ASR model.")
            
        logger.info(f"Loading ASR Model: {self.model_name}...")
        try:
            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=self.model_name)
            
            # Put on GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
            self.is_loaded = True
            logger.info(f"ASR Model {self.model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ASR Model: {e}")
            raise
            
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes a 16kHz mono WAV file into text.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("ASR Model is not loaded.")
            
        try:
            # NeMo's transcribe method takes a list of audio file paths
            # Note: For RNNT models, it returns a tuple of lists, we just need the text
            transcriptions = self.model.transcribe([audio_path])
            
            # Extract the string depending on what it returns.
            # Usually transcribe() returns a list of strings, or a tuple where the first element is the text list.
            if isinstance(transcriptions, tuple):
                transcription_list = transcriptions[0]
            else:
                transcription_list = transcriptions
                
            if transcription_list and len(transcription_list) > 0:
                item = transcription_list[0]
                if hasattr(item, 'text'):
                    # NeMo Hypothesis object
                    return item.text.strip()
                elif isinstance(item, str):
                    return item.strip()
                elif isinstance(item, dict) and 'text' in item:
                    return item['text'].strip()
                return str(item).strip()
            return ""
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
