import torch
import os
import gdown

import json
import glob
import torchaudio
import IPython.display as ipd
from functions import *
import matplotlib.pyplot as plt

if not os.path.exists('EfficientConformer'):
  os.mkdir('EfficientConformer')
os.chdir('EfficientConformer/')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# choose pretrained model
pretrained_models = {
    "EfficientConformerCTCSmall": "1MU49nbRONkOOGzvXHFDNfvWsyFmrrBam",
    "EfficientConformerCTCMedium": "1h5hRG9T_nErslm5eGgVzqx7dWDcOcGDB",
    "EfficientConformerCTCLarge": "1U4iBTKQogX4btE-S4rqCeeFZpj3gcweA"
}
pretrained_model = "EfficientConformerCTCSmall"
if not os.path.exists(os.path.join('callbacks', pretrained_model)):
  os.makedirs(os.path.join('callbacks', pretrained_model))

# download pretrained model
gdown.download("https://drive.google.com/uc?id=" + pretrained_models[pretrained_model], os.path.join("callbacks", pretrained_model, "checkpoints_swa-equal-401-450.ckpt"), quiet=False)

# download pretrained model tokenizer
if not os.path.exists(os.path.join('datasets', 'LibriSpeech')):
  os.makedirs(os.path.join('datasets', 'LibriSpeech'))
gdown.download("https://drive.google.com/uc?id=1hx2s4ZTDsnOFtx5_h5R_qZ3R6gEFafRx", "datasets/LibriSpeech/LibriSpeech_bpe_256.model", quiet=False)

# create and read in pretrained model
# config_file = pretrained_model + '.json'
config_file = "configs/" + pretrained_model + ".json"
config_file = "/media/huzq85/2TB_Work/2-working/efficient_conformer/configs/" + pretrained_model + ".json"
with open(config_file) as json_config:
  config = json.load(json_config)

model = create_model(config).to(device)
model.eval()
model.load(os.path.join("callbacks", pretrained_model, "checkpoints_swa-equal-401-450.ckpt"))
  
# Get audio files paths
audio_files = glob.glob("datasets/LibriSpeech/*/*/*/*.flac")
print(len(audio_files), "audio files")

# Random indices
indices = torch.randint(0, len(audio_files), size=(10,))
# Test model
for i in indices:

  # Load audio file
  audio, sr = torchaudio.load(audio_files[i])

  # Plot audio
  plt.title(audio_files[i].split("/")[-1])
  plt.plot(audio[0])
  plt.show()
  print()

  # Display
  ipd.display(ipd.Audio(audio, rate=sr))
  print()

  # Predict sentence
  prediction = model.gready_search_decoding(audio.to(device), x_len=torch.tensor([len(audio[0])], device=device))[0]
  print("model prediction:", prediction, '\n')

  for i in range(100):
    print('*', end='')
  print('\n')