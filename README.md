# 🎤 Koach Core - Korean Pronunciation Analysis Engine

> **한국어 발음 교정을 위한 핵심 분석 엔진**

---

## 📖 개요

**Koach Core**는 한국어 발음 교정을 위한 핵심 분석 엔진입니다. 음성 인식, 음성 정렬, 발음 분석, 그리고 AI 기반 피드백 생성을 통해 외국어 학습자의 한국어 발음을 정밀하게 분석하고 개선점을 제공합니다.

## 🚀 주요 기능

### 🔍 핵심 분석 기능
- **음성 인식**: Whisper 모델을 활용한 정확한 한국어 음성-텍스트 변환
- **강제 정렬**: Montreal Forced Alignment(MFA)를 통한 음소 단위 시간 정보 추출
- **발음 분석**: 음소별 정확도 및 오류 패턴 분석
- **억양 분석**: Pitch, 지속시간, 강세 패턴 분석
- **참조 비교**: 원어민 발음과의 정량적 비교 분석

### 🤖 AI 피드백
- **GPT 기반 피드백**: OpenAI GPT-4를 활용한 자연어 피드백 생성
- **RAG 지식베이스**: 한국어 발음 교육 지식을 활용한 향상된 피드백
- **개인화된 조언**: 학습자 수준에 맞는 맞춤형 개선 방안 제시

### 📊 시각화
- **음성 파형 분석**: 학습자와 원어민 발음의 시각적 비교
- **Pitch 곡선**: 억양 패턴의 그래프 표현
- **음소 정확도 차트**: 발음 정확도의 시각적 표현

---

## 🏗️ 프로젝트 구조

```
koach/
├── main.py                 # 메인 실행 파일
├── core/                   # 핵심 분석 엔진
│ ├── koach.py              # 메인 분석 클래스
│ ├── prosody.py            # 억양 분석 모듈
│ └── knowledge_base.py     # RAG 지식베이스
├── config/ # 설정 관리
│ └── settings.py           # 설정 파일
├── utils/ # 유틸리티 함수
│ ├── audio.py              # 오디오 처리
│ └── text.py               # 텍스트/음성 정렬
├── models/                 # 사전 훈련 모델
│ ├── korean_mfa.zip        # 한국어 MFA 음성 모델
│ ├── korean_mfa.dict       # 한국어 MFA 사전
│ └── whisper/              # Whisper 모델 캐시
├── knowledge/              # RAG 지식베이스
├── temp/                   # 임시 파일 (중간 결과물 저장)
└── README.md               # 이 파일
```

---

## ⚙️ 설치 및 설정

### 1. 필수 요구사항

- **Python 3.8+**
- **FFmpeg** (오디오 변환용)
- **Montreal Forced Alignment (MFA)** (음성 정렬용)
- **OpenAI API Key** (AI 피드백용)

### 2. 의존성 설치

```bash
# 프로젝트 루트에서 실행
pip install -r requirements.txt
```

### 3. MFA 설치 및 설정

```bash
# MFA 설치
conda install -c conda-forge montreal-forced-alignment

# 한국어 모델 다운로드
mfa model download acoustic korean_mfa
mfa model download dictionary korean_mfa
```

### 4. 환경 변수 설정

```bash
# .env 파일 생성 (프로젝트 루트에)
echo "OPENAI_API_KEY=your_openai_api_key_here" > ../.env
```

---

## 🎯 사용법

### 1. 명령행 인터페이스

```bash
# 기본 사용법
python main.py learner_audio.wav native_audio.wav "발음할 텍스트"

# 상세 옵션
python main.py \
    --file input/learner.m4a \
    --reference input/native.wav \
    --text "안녕하세요" \
    --output-dir results/ \
    --model-size base
```

### 2. Python 스크립트에서 사용

```python
from core.koach import Koach

# Koach 인스턴스 생성
koach = Koach(config={
    "whisper_model": "base",
    "use_rag": True,
    "openai_model": "gpt-4o"
})

# 발음 분석 실행
result = koach.analyze_pronunciation(
    learner_audio="path/to/learner.wav",
    native_audio="path/to/native.wav",
    script="안녕하세요",
    visualize=True
)

# 결과 확인
print(f"유사도 점수: {result['similarity_score']}")
print(f"AI 피드백: {result['feedback']}")
```

### 3. 명령행 옵션

| 옵션 | 설명 | 기본값 |
|------|------|---------|
| `--file, -f` | 학습자 음성 파일 | - |
| `--reference, -r` | 원어민 참조 음성 | - |
| `--text, -t` | 목표 발음 텍스트 | - |
| `--output-dir, -o` | 결과 저장 디렉토리 | `output/` |
| `--model-size, -m` | Whisper 모델 크기 | `base` |
| `--no-rag` | RAG 지식베이스 비활성화 | False |
| `--no-visualization` | 시각화 비활성화 | False |
| `--quiet, -q` | 최소 출력 모드 | False |

---

## 📊 출력 결과

### 1. 분석 결과 구조

```python
{
    "similarity_score": 0.85,           # 전체 유사도 점수 (0-1)
    "feedback": "AI 생성 피드백 텍스트",
    "phoneme_analysis": {               # 음소별 분석
        "accuracy": 0.9,
        "errors": [...]
    },
    "prosody_analysis": {               # 억양 분석
        "pitch_similarity": 0.8,
        "rhythm_score": 0.75
    },
    "visualization_paths": [            # 생성된 시각화 파일 경로
        "output/phoneme_accuracy.png",
        "output/pitch_comparison.png"
    ]
}
```

### 2. 생성되는 파일

```
data/output/
└── analysis_result.json     # 상세 분석 결과 (GPT 결과 포함)
```

---

## 🔧 설정 옵션

### 1. 기본 설정 (`config/settings.py`)

```python
CURRENT_CONFIG = {
    "whisper": {
        "model": "base",
        "language": "ko"
    },
    "openai": {
        "model": "gpt-4o",
        "temperature": 0.3
    },
    "mfa": {
        "acoustic_model": "korean_mfa",
        "dictionary": "korean_mfa"
    }
}
```

### 2. 사용자 정의 설정

```python
custom_config = {
    "whisper_model": "large",       # 더 정확한 모델 사용
    "use_rag": False,               # RAG 비활성화
    "openai_model": "gpt-3.5-turbo" # 더 빠른 모델 사용
}

koach = Koach(config=custom_config)
```

---

## 🧠 기술 스택

| 구성 요소 | 기술 | 용도 |
|-----------|------|------|
| **음성 인식** | OpenAI Whisper | 음성-텍스트 변환 |
| **음성 정렬** | Montreal Forced Alignment | 음소 단위 정렬 |
| **AI 피드백** | OpenAI GPT-4 | 자연어 피드백 생성 |
| **지식베이스** | FAISS + Sentence Transformers | RAG 시스템 |
| **오디오 처리** | librosa, pydub | 음성 신호 처리 |
| **시각화** | matplotlib, seaborn | 결과 시각화 |

---

## 🔍 성능 벤치마크

### 1. 음성 인식 정확도
- **한국어 단문**: 95%+
- **한국어 복문**: 90%+
- **외국인 발음**: 85%+

### 2. 음성 정렬 정확도
- **원어민 발음**: 98%+
- **학습자 발음**: 92%+

### 3. 처리 속도
- **1분 음성 분석**: 30-60초
- **Whisper 전사**: 10-20초
- **MFA 정렬**: 15-30초
- **AI 피드백 생성**: 5-10초

---

## 🚨 문제 해결

### 1. 일반적인 오류

#### MFA 설치 문제
```bash
# Conda 환경에서 MFA 재설치
conda install -c conda-forge montreal-forced-alignment=2.2.17
```

#### CUDA 메모리 부족
```python
# 설정에서 모델 크기 줄이기
config = {"whisper_model": "tiny"}  # base 대신 tiny 사용
```

#### API 키 오류
```bash
# .env 파일 확인
cat ../.env
# OPENAI_API_KEY=your_key_here 형식인지 확인
```

### 2. 성능 최적화

#### 빠른 분석을 위한 설정
```python
fast_config = {
    "whisper_model": "tiny",
    "use_rag": False,
    "visualization": False
}
```

#### 정확한 분석을 위한 설정
```python
accurate_config = {
    "whisper_model": "large",
    "use_rag": True,
    "openai_model": "gpt-4"
}
```

---

## 🔄 업데이트 로그

### v1.0.0 (Current)
- ✅ 완전한 음성 인식 및 정렬 파이프라인
- ✅ GPT-4 기반 AI 피드백 시스템
- ✅ RAG 지식베이스 통합
- ✅ 종합적인 시각화 기능
- ✅ CLI 및 Python API 지원

---

## 📞 문의 및 지원

- **이슈 리포트**: GitHub Issues
- **기능 요청**: GitHub Discussions
- **기술 문의**: [contact@koach.ai](mailto:contact@koach.ai)

---

## 📜 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

---

> **Koach Core는 정확하고 효과적인 한국어 발음 교정을 위한 강력한 분석 엔진입니다.**
