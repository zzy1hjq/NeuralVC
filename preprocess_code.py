import torch
import os
import librosa

hubert = torch.hub.load("bshall/hubert:main", f"hubert_soft").eval() 

def get_code(vctk_path):
    speakers = os.listdir(vctk_path)
    for spk in speakers:
        files_path = f"{vctk_path}/{spk}"
        wavs_path = os.listdir(files_path)
        for wav_name in wavs_path:
            wav_path = f"{files_path}/{wav_name}"
            wav, r = librosa.load(wav_path)
            wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
            c = hubert.units(wav)
            c = c.transpose(1,2)
            torch.save(c, wav_path.replace(".wav", ".pt"))
            c_path = wav_path.replace(".wav", ".pt")
            print(f"content code saved in {c_path}")

if __name__ == "__main__":
    vctk_path = ".dataset/vctk-16k"
    get_code(vctk_path)