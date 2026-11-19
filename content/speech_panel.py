"""Microphone/upload/display UI only. The notebook supplies transcription."""
import asyncio
import base64
import json
from pathlib import Path
import secrets

import numpy as np
import pyjs
from IPython.display import IFrame, display


class SpeechPanel:
    def __init__(self, transcribe, *, sample, device, seconds=15):
        self.transcribe = transcribe
        self.closed = False
        self.cancelled = False
        self.busy = False
        self.tasks = set()
        self.history = []
        name = 'whisper-speech-' + secrets.token_hex(12)
        callback, self.callback_owner = pyjs.create_callable(self._receive)
        self.channel = pyjs.js.eval('''(name, callback) => {
            const channel = new BroadcastChannel(name);
            channel.onmessage = ({data}) => callback(JSON.stringify(data));
            return {
                postMessage: message => channel.postMessage(message),
                close: () => {channel.onmessage = null; channel.close();}
            };
        }''')(name, callback)
        payload = dict(channel=name, device=device, maxSeconds=seconds,
                       sample=base64.b64encode(Path(sample).read_bytes()).decode())
        self.max_samples = seconds * 16000
        html = Path(__file__).with_name('speech_panel.html').read_text()
        html = html.replace('__SPEECH_PAYLOAD__', json.dumps(payload))
        self.url = str(pyjs.js.eval(
            "html => URL.createObjectURL(new Blob([html], {type:'text/html'}))")(html))
        display(IFrame(self.url, width='100%', height=730,
                       extras=['allow="microphone"', 'title="Whisper speech to text"',
                               'style="border:0;border-radius:12px"']))

    def _send(self, message):
        if not self.closed:
            self.channel.postMessage(pyjs.js.JSON.parse(json.dumps(message)))

    def _receive(self, text):
        if self.closed:
            return
        request = json.loads(str(text))
        if request.get('kind') == 'cancel':
            self.cancelled = True
            return
        if request.get('kind') != 'transcribe':
            return
        if self.busy:
            self._send(dict(kind='result', id=request.get('id'), error='A clip is already running.'))
            return
        self.busy = True
        self.cancelled = False
        task = asyncio.create_task(self._respond(request))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _respond(self, request):
        try:
            encoded = request.get('pcm', '')
            if not isinstance(encoded, str) or len(encoded) > (self.max_samples * 4 + 2) * 4 // 3 + 4:
                raise ValueError('Audio clip exceeds the recording limit.')
            audio = np.frombuffer(base64.b64decode(encoded, validate=True), dtype='<f4').copy()
            if not 1 <= len(audio) <= self.max_samples or not np.isfinite(audio).all():
                raise ValueError('Expected finite mono audio samples at 16 kHz.')
            result = await self.transcribe(
                audio, language=request.get('language', 'en'),
                on_partial=lambda text: self._send(dict(kind='partial', id=request['id'], text=text)),
                cancelled=lambda: self.cancelled or self.closed,
            )
            self.history.append({k: v for k, v in result.items() if k != 'token_ids'})
            del self.history[:-30]
            self._send(dict(result, kind='result', id=request['id']))
        except Exception as error:
            self._send(dict(kind='result', id=request.get('id'), error=str(error)))
        finally:
            self.busy = False

    async def close(self):
        if self.closed:
            return
        self._send(dict(kind='closed'))
        self.closed = True
        self.cancelled = True
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
        self.channel.close()
        self.callback_owner.delete()
        pyjs.js.URL.revokeObjectURL(self.url)
