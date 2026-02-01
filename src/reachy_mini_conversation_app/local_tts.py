#!/usr/bin/env python3
""" client for WebSocket TTS endpoint with realistic LLM token streaming."""

import asyncio
import json
import re
import time
import wave
from collections import deque
from reachy_mini_conversation_app.config import config

import logging
logger = logging.getLogger(__name__)

# Magpie outputs at 22kHz
MAGPIE_SAMPLE_RATE = 22000

# Regex pattern for emoji and other non-speakable characters
# Covers most emoji ranges including emoticons, symbols, and pictographs
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
    "]+",
    flags=re.UNICODE
)

_segment_sentence_boundary_queue: deque[bool] = deque()

def sanitize_text_for_tts(text: str) -> str:
    """Remove emojis and normalize special characters for TTS.

    Args:
        text: Raw text from LLM

    Returns:
        Sanitized text safe for TTS synthesis
    """
    # Remove emojis
    text = EMOJI_PATTERN.sub("", text)
    # Normalize curly quotes and dashes
    text = text.replace("\u2018", "'")  # LEFT SINGLE QUOTATION MARK
    text = text.replace("\u2019", "'")  # RIGHT SINGLE QUOTATION MARK
    text = text.replace("\u201C", '"')  # LEFT DOUBLE QUOTATION MARK
    text = text.replace("\u201D", '"')  # RIGHT DOUBLE QUOTATION MARK
    text = text.replace("\u2014", "-")  # EM DASH
    text = text.replace("\u2013", "-")  # EN DASH
    return text


# Sentence boundary pattern - matches .!? followed by optional quotes/parens and space
SENTENCE_BOUNDARY_PATTERN = re.compile(r'([.!?]["\'\)]*\s)')


def split_into_sentences(text: str) -> list[str]:
    """Split text into individual sentences for TTS.

    Splits at sentence boundaries (.!? followed by space) to keep TTS chunks
    small and avoid GPU OOM on long text. Each sentence retains its trailing
    whitespace for proper TTS pacing.

    Args:
        text: Text containing one or more sentences

    Returns:
        List of sentences. If no sentence boundaries found, returns [text].

    Example:
        >>> split_into_sentences("Hello! How are you? I'm fine. ")
        ["Hello! ", "How are you? ", "I'm fine. "]
    """
    if not text:
        return []

    # Split on sentence boundaries, keeping the delimiter
    parts = SENTENCE_BOUNDARY_PATTERN.split(text)

    # Recombine: each sentence = content + delimiter
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and SENTENCE_BOUNDARY_PATTERN.match(parts[i + 1]):
            # Content followed by delimiter
            sentences.append(parts[i] + parts[i + 1])
            i += 2
        elif parts[i]:
            # Trailing content without delimiter (incomplete sentence)
            sentences.append(parts[i])
            i += 1
        else:
            i += 1

    return sentences if sentences else [text] if text else []

def _generate_silence_frames( duration_ms: int) -> bytes:
        """Generate silence audio (zeros) for the given duration.

        Args:
            duration_ms: Duration of silence in milliseconds

        Returns:
            Bytes of silent audio (16-bit PCM, mono, at sample_rate)
        """
        # 16-bit PCM = 2 bytes per sample
        num_samples = int(MAGPIE_SAMPLE_RATE * duration_ms / 1000)
        return bytes(num_samples * 2)


try:
    from websockets.asyncio.client import connect as ws_connect
except ImportError:
    from websockets import connect as ws_connect

def _ends_at_sentence_boundary(text: str) -> bool:
        """Check if text ends at a sentence boundary (for inter-sentence pauses)."""
        text = text.strip()
        return bool(text) and text[-1] in '.!?'


async def tts(text):
    ws_url = config.TTS_URL

    audio_chunks = []
    
    async with await ws_connect(ws_url) as ws:
        logger.info("WS Connected!")

        # Send init message
        await ws.send(json.dumps({
            "type": "init",
            "voice": "aria",
            "language": "en"
        }))

        # Wait for stream_created
        response = await ws.recv()
        data = json.loads(response)

        # Background task to receive audio
        async def receive_audio():
            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        logger.info("audio bytes received")
                        audio_chunks.append(message)
                    else:
                        data = json.loads(message)
                        if data.get("type") == "done":
                            logger.info(f"\n>>> Stream done: {data.get('total_audio_ms', 0):.0f}ms audio")
                            _segment_sentence_boundary_queue.clear()
                            break
                        elif data.get("type") == "segment_complete":
                            logger.info("audio segement completed")
                            # Inject silence pause if this segment ended at sentence boundary
                            if _segment_sentence_boundary_queue:
                                ended_with_sentence = _segment_sentence_boundary_queue.popleft()
                                if ended_with_sentence:
                                    silence = _generate_silence_frames(
                                         250
                                    )
                                    audio_chunks.append(silence)
                     
                            
            except Exception as e:
                logger.error(f"\nReceive error: {e}")

        # Start receiver task
        receiver = asyncio.create_task(receive_audio())

        sentences = split_into_sentences(text)
        for sentence in sentences:
            if not sentence or not sentence.strip():
                continue

            # Build text message with mode selection
            msg = {"type": "text", "text": sentence}
            msg["mode"] = "batch"

            await ws.send(json.dumps(msg))
            _segment_sentence_boundary_queue.append(
                    _ends_at_sentence_boundary(sentence)
                )
        
        await ws.send(json.dumps({"type": "close"}))
        
        # Wait for receiver to complete
        await receiver

    return audio_chunks


if __name__ == "__main__":
    asyncio.run(test_websocket_tts())