"""5단계 음성 테스트: 텍스트 질의로 소리 찾기. 입력은 16 kHz 모노만 받는다.

샘플은 scripts/make_audio_samples.sh 로 만든다 (speech.wav, beep.wav, noise.wav).
"""
import soundfile as sf

from bench import Recorder, media_dir
from eg2 import config_name, load

# (질의, 정답 파일 이름)
QUERIES = [("덥다고 냉방을 요청하는 목소리", "speech.wav"),
           ("날카로운 경보음", "beep.wav"),
           ("웅웅거리는 배경 소음", "noise.wav")]


def main():
    media = media_dir()
    files = sorted(str(p) for p in media.glob("*.wav") if not p.name.startswith("raw_"))
    if not files:
        raise SystemExit(f"{media} 에 wav 가 없습니다. scripts/make_audio_samples.sh 를 먼저 실행하세요")
    for f in files:
        info = sf.info(f)
        # 샘플레이트·채널이 다르면 검색 결과가 무작위에 가까워진다 (자료 6단계)
        assert info.samplerate == 16000 and info.channels == 1, f"{f}: 16kHz 모노가 아님"

    rec = Recorder("05_audio", config_name(audio=True))
    model = load(audio=True)
    rec.loaded()

    inputs = [{"audio": f} for f in files]
    model.encode(inputs[:1], normalize_embeddings=True)   # 데우기
    with rec.time(f"WAV {len(files)}개", n=len(files)):
        aud_emb = model.encode(inputs, normalize_embeddings=True)

    q = model.encode([qi for qi, _ in QUERIES], prompt_name="SearchQuery", normalize_embeddings=True)
    names = [f.split("/")[-1] for f in files]
    for (qi, answer), row in zip(QUERIES, q @ aud_emb.T):
        best = int(row.argmax())
        rec.check(f"'{qi}'", names[best] == answer, f"-> {names[best]} ({row[best]:.3f})")
    rec.save()


if __name__ == "__main__":
    main()
