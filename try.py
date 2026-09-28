import torch
import torchaudio
from transformers import AutoConfig, AutoModel, AutoFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRAM():
    def __init__(self):
        self.model = AutoModel.from_pretrained("labhamlet/gramt-binaural-time", trust_remote_code=True).to(device)
        self.extractor = AutoFeatureExtractor.from_pretrained("labhamlet/gramt-binaural-time", trust_remote_code=True, device=device)
        for param in self.model.parameters():
            param.requires_grad = False

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000,
            n_fft=1024,
            win_length=1024,
            hop_length=320,
            f_min=50,
            f_max=32000 // 2,
            n_mels=128,
            power=2.0,
        ).to(device)


    def embed(self, audio):
        audio = audio.to(device)
        if audio.ndim < 3:
            audio = audio.unsqueeze(0)
        mel = self.melspec(audio).transpose(3,2)
        logmel = (mel + torch.finfo().eps).log().to(device)
        with torch.no_grad():
            output = self.model(logmel, strategy='raw')
        return output

gram = GRAM()

n_samples = int(0.128 * 32000)  # ~4096 samples
audio = torch.randn(1, 2, n_samples)  # (batch, channels, time)

new_obs_embedded = gram.embed(audio).to(device)  # 128ms of Fs=32000, binaural audio
print("Embedding shape:", new_obs_embedded.shape)