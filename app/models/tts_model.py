import io
import base64
import numpy as np
import soundfile as sf
import logging

logger = logging.getLogger(__name__)

class KokoroTTSModel:
    def __init__(self, lang_code='a'):
        self.lang_code = lang_code
        self.pipeline = None
        self.is_loaded = False
        
    def load(self):
        if self.is_loaded:
            return
        logger.info("Initializing Kokoro TTS Pipeline...")
        try:
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code=self.lang_code, device='cpu')
            self.is_loaded = True
            logger.info("Kokoro TTS initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load Kokoro TTS: {e}")
            raise
            
    def synthesize_base64(self, text: str, voice: str = 'af_heart') -> str:
        """
        Synthesizes text into speech and returns a base64 encoded WAV string.
        """
        if not self.is_loaded or not self.pipeline:
            raise RuntimeError("Kokoro TTS model is not loaded.")
            
        generator = self.pipeline(text, voice=voice, speed=1.0)
        all_audio = []
        for i, (gs, ps, audio) in enumerate(generator):
            all_audio.append(audio)
            
        if not all_audio:
            logger.warning("Kokoro TTS generated empty audio sequence.")
            return ""
            
        full_audio = np.concatenate(all_audio)
        
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, 24000, format='WAV')
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')
