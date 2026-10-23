"""5단계 자료 만들기 (ffmpeg 없이): 경보음·잡음 생성, 합성 음성을 16 kHz 모노로 변환.

nvidia-jetpack 이 설치한 NVIDIA 판 ffmpeg 에는 lavfi(sine, anoisesrc)가 빠져 있어
ffmpeg 로는 경보음·잡음을 만들 수 없다. 그래서 venv 의 numpy·soundfile·librosa 로 만든다.
사용: python make_audio_samples.py   (합성 음성 raw_speech.wav 는 스크립트가 espeak-ng 로 먼저 만든다)
"""
import numpy as np
import soundfile as sf

from bench import media_dir

SR = 16000


def to_16k_mono(src, dst):
    import librosa  # 무거운 import 는 변환할 때만
    y, _ = librosa.load(src, sr=SR, mono=True)
    sf.write(dst, y, SR, subtype="PCM_16")


def pink_noise(n, rng):
    # 백색 잡음 스펙트럼에 1/sqrt(f) 를 곱해 핑크 잡음을 만든다 (ffmpeg anoisesrc=pink 와 같은 성질)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec))
    spec[1:] /= np.sqrt(f[1:])
    y = np.fft.irfft(spec, n)
    return y / np.abs(y).max() * 0.5


def main():
    media = media_dir()
    media.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)   # 같은 잡음이 다시 나오게 고정
    t = np.arange(3 * SR) / SR
    sf.write(media / "beep.wav", 0.5 * np.sin(2 * np.pi * 1000 * t), SR, subtype="PCM_16")
    sf.write(media / "noise.wav", pink_noise(5 * SR, rng), SR, subtype="PCM_16")
    for raw in sorted(media.glob("raw_*.wav")):
        to_16k_mono(raw, media / raw.name[len("raw_"):])
    for p in sorted(media.glob("*.wav")):
        if not p.name.startswith("raw_"):
            info = sf.info(p)
            print(f"{p.name}: {info.samplerate} Hz, {info.channels}ch, {info.duration:.1f}s")


if __name__ == "__main__":
    main()
