# 🔍 Koach 프롬프트 디버깅 가이드

프롬프트 엔지니어링과 디버깅을 위한 종합 가이드입니다.

## 🚀 빠른 시작

### 1. 기본 프롬프트 확인
```bash
# 터미널에 프롬프트 출력
python main.py input/learner.m4a input/native.m4a "안녕하세요" --show-prompt

# 프롬프트만 생성 (GPT 호출 없이)
python main.py input/learner.m4a input/native.m4a "안녕하세요" --prompt-only
```

### 2. 상세 디버깅 모드
```bash
# 프롬프트를 파일로 저장하고 디버깅 정보 출력
python main.py input/learner.m4a input/native.m4a "안녕하세요" --debug-prompt
```

### 3. 조합 사용
```bash
# 모든 디버깅 옵션 활성화
python main.py input/learner.m4a input/native.m4a "안녕하세요" \
  --debug-prompt --show-prompt
```

## 📁 저장되는 파일들

프롬프트 디버깅이 활성화되면 `koach/temp/debug/` 폴더에 다음 파일들이 생성됩니다:

### 프롬프트 파일 (텍스트)
```
koach/temp/debug/prompt_20241212_143052.txt
```
- 생성된 프롬프트의 전체 내용
- 타임스탬프와 메타데이터 포함
- 가독성을 위한 텍스트 형식

### 디버그 JSON 파일
```
koach/temp/debug/debug_20241212_143052.json
```
- 프롬프트와 입력 데이터
- 분석 결과 요약
- 시스템 설정 정보

## 🔍 디버깅 출력 예시

### 터미널 출력
```
🔍 프롬프트 디버그 정보 저장 완료
📄 프롬프트 파일: koach/temp/debug/prompt_20241212_143052.txt
📊 디버그 JSON: koach/temp/debug/debug_20241212_143052.json
📝 프롬프트 길이: 1,247 문자
📋 프롬프트 미리보기: 당신은 친절한 한국어 발음 교정 전문가입니다...

🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍
🔍 PROMPT DEBUG
🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍
📝 프롬프트 길이: 1247 문자
📊 사용된 모델: gpt-4o
🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍

📋 생성된 프롬프트:
--------------------------------------------------------------------------------
당신은 친절한 한국어 발음 교정 전문가입니다.

**학습 목표 문장**: 안녕하세요

**분석 데이터**:
- 학습자 발화: 안녕하세요
- 원어민 발화: 안녕하세요
- 학습자 타이밍: ㅏ(0.2s) | ㄴ(0.1s) | ㄴ(0.15s) | ㅕ(0.18s) | ㅇ(0.12s)...
...
--------------------------------------------------------------------------------

💾 최신 프롬프트 파일: koach/temp/debug/prompt_20241212_143052.txt
📖 파일 내용 보기:
    cat 'koach/temp/debug/prompt_20241212_143052.txt'
📊 JSON 디버그 파일 보기:
    cat 'koach/temp/debug/debug_20241212_143052.json'
```

### JSON 디버그 파일 구조
```json
{
  "timestamp": "20241212_143052",
  "prompt": "당신은 친절한 한국어 발음 교정 전문가입니다...",
  "analysis_input": {
    "learner_text": "안녕하세요",
    "native_text": "안녕하세요",
    "script_text": "안녕하세요",
    "learner_timing_preview": "ㅏ(0.2s) | ㄴ(0.1s) | ㄴ(0.15s)...",
    "native_timing_preview": "ㅏ(0.18s) | ㄴ(0.12s) | ㄴ(0.14s)..."
  },
  "prosody_data": {
    "phoneme_count": 15,
    "prosody_available": true,
    "comparison_available": true
  },
  "config": {
    "model": "gpt-4o",
    "use_rag": true,
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
  }
}
```

## 🛠️ 고급 디버깅 기법

### 1. 프롬프트 비교 분석
```bash
# 서로 다른 입력으로 프롬프트 생성
python main.py input/audio1.m4a input/native.m4a "안녕" --debug-prompt
python main.py input/audio2.m4a input/native.m4a "안녕하세요" --debug-prompt

# 생성된 프롬프트 비교
diff koach/temp/debug/prompt_*.txt
```

### 2. JSON 데이터 분석
```bash
# jq를 사용한 JSON 분석
cat koach/temp/debug/debug_*.json | jq '.prosody_data'
cat koach/temp/debug/debug_*.json | jq '.analysis_input.learner_text'
```

### 3. 프롬프트 길이 통계
```bash
# 프롬프트 길이 확인
cat koach/temp/debug/debug_*.json | jq '.prompt | length'

# 평균 길이 계산
cat koach/temp/debug/debug_*.json | jq '.prompt | length' | awk '{sum+=$1} END {print "Average:", sum/NR}'
```

## 🌟 LangSmith 연동 (선택적)

### 환경 변수 설정
```bash
export LANGSMITH_API_KEY="your_langsmith_api_key"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="koach-pronunciation-analysis"
```

### 사용법
```bash
# LangSmith 트레이싱과 함께 실행
python main.py input/learner.m4a input/native.m4a "안녕하세요" --debug-prompt
```

LangSmith에서 다음 정보를 확인할 수 있습니다:
- 프롬프트 전체 내용
- GPT 응답 결과
- 토큰 사용량
- 응답 시간
- 체인 추적

## 📊 프롬프트 최적화 팁

### 1. 길이 최적화
- 현재 평균 프롬프트 길이: ~1,200-1,500자
- 토큰 제한을 고려한 적절한 길이 유지
- 핵심 정보 우선 포함

### 2. 구조 개선
- 명확한 섹션 구분 (`**`, `##` 사용)
- 우선순위에 따른 정보 배치
- 일관된 형식 유지

### 3. 컨텍스트 조정
- RAG 검색 결과 활용도 모니터링
- 운율 분석 데이터의 유효성 확인
- 불필요한 정보 제거

## 🔧 트러블슈팅

### 프롬프트가 저장되지 않는 경우
```bash
# 권한 확인
ls -la koach/temp/
chmod 755 koach/temp/

# 디렉토리 수동 생성
mkdir -p koach/temp/debug
```

### 프롬프트가 너무 긴 경우
- `max_length` 파라미터 조정
- 타이밍 정보 요약 수준 변경
- RAG 결과 수 줄이기 (`top_k` 감소)

### LangSmith 연동 문제
```bash
# 환경 변수 확인
echo $LANGSMITH_API_KEY
echo $LANGCHAIN_TRACING_V2

# 연결 테스트
curl -H "Authorization: Bearer $LANGSMITH_API_KEY" https://api.smith.langchain.com/
```

## 📝 실제 사용 예시

### 시나리오 1: 프롬프트 엔지니어링
```bash
# 기본 프롬프트 확인
python main.py test_audio.m4a reference.m4a "테스트 문장" --prompt-only --show-prompt

# 수정 후 다시 테스트
python main.py test_audio.m4a reference.m4a "테스트 문장" --debug-prompt
```

### 시나리오 2: 성능 분석
```bash
# 여러 테스트 케이스 실행
for audio in test_cases/*.m4a; do
    python main.py "$audio" reference.m4a "테스트" --debug-prompt --quiet
done

# 결과 분석
cat koach/temp/debug/debug_*.json | jq '.config.model'
```

이 가이드를 통해 Koach의 프롬프트를 효과적으로 디버깅하고 최적화할 수 있습니다! 🚀 