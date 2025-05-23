# -*- coding: utf-8 -*-
"""
프롬프트 템플릿 관리
====================

다양한 상황에 맞는 프롬프트 템플릿을 중앙에서 관리합니다.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """프롬프트 타입 정의"""
    HIGH_QUALITY = "high_quality"
    MEDIUM_QUALITY = "medium_quality"
    BASIC = "basic"
    MINIMAL = "minimal"


@dataclass
class PromptTemplate:
    """프롬프트 템플릿 구조"""
    name: str
    template: str
    description: str
    max_tokens: int
    required_fields: list
    optional_fields: list


class PromptTemplateManager:
    """프롬프트 템플릿 관리자"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """프롬프트 템플릿 로드"""
        return {
            PromptType.HIGH_QUALITY.value: PromptTemplate(
                name="고품질 분석",
                template="""당신은 한국어 발음 교정 전문가입니다.

**목표**: {script_text}

**분석 데이터**:
- 학습자: {learner_text}
- 원어민: {native_text}
- 학습자 타이밍: {learner_timing}
- 원어민 타이밍: {native_timing}{prosody_context}{rag_context}

**분석 요청**:
1. 발음 오류 (단어/음소별, 한글 설명)
2. 차이점 분석 (억양, 강세, 속도)
3. 교정 방법 (구체적)
4. 연습법 제안

**응답 형식**:
## 📊 분석
[오류사항 - 구체적 단어/음소]

## 🎯 교정
[발음/억양/강세 개선법]

## 💡 연습
[실용적 연습법]

## ⭐️ 격려
[잘한 점과 동기부여]

간결하고 실용적으로 답하세요.""",
                description="모든 데이터가 있을 때 사용하는 상세 분석 템플릿",
                max_tokens=1500,
                required_fields=["script_text", "learner_text", "native_text"],
                optional_fields=["learner_timing", "native_timing", "prosody_context", "rag_context"]
            ),
            
            PromptType.MEDIUM_QUALITY.value: PromptTemplate(
                name="중품질 분석",
                template="""한국어 발음 교정 전문가로서 분석해주세요.

**목표**: {script_text}
**학습자**: {learner_text}
**원어민**: {native_text}{prosody_context}

**분석 요청**:
1. 주요 발음 차이점
2. 개선 방법
3. 연습법

**응답**:
## 분석
[핵심 차이점]

## 개선법
[구체적 방법]

## 연습
[실용적 팁]

간결하게 답하세요.""",
                description="기본 데이터만 있을 때 사용하는 간소화 템플릿",
                max_tokens=800,
                required_fields=["script_text", "learner_text", "native_text"],
                optional_fields=["prosody_context"]
            ),
            
            PromptType.BASIC.value: PromptTemplate(
                name="기본 분석",
                template="""한국어 발음을 비교 분석해주세요.

목표: {script_text}
학습자: {learner_text}
원어민: {native_text}

주요 차이점과 개선 방법을 간단히 설명해주세요.""",
                description="최소 데이터로 빠른 분석",
                max_tokens=400,
                required_fields=["script_text", "learner_text", "native_text"],
                optional_fields=[]
            ),
            
            PromptType.MINIMAL.value: PromptTemplate(
                name="최소 분석",
                template="""다음 한국어 발음을 비교하고 개선점을 제안해주세요:

학습자: {learner_text}
원어민: {native_text}

2-3가지 핵심 개선점만 간단히 답하세요.""",
                description="응급 상황용 초간단 템플릿",
                max_tokens=200,
                required_fields=["learner_text", "native_text"],
                optional_fields=[]
            )
        }
    
    def get_template(self, template_type: PromptType) -> PromptTemplate:
        """템플릿 가져오기"""
        return self.templates.get(template_type.value)
    
    def format_prompt(
        self, 
        template_type: PromptType, 
        **kwargs
    ) -> str:
        """템플릿 포맷팅"""
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"템플릿을 찾을 수 없습니다: {template_type}")
        
        # 필수 필드 확인
        for field in template.required_fields:
            if field not in kwargs:
                raise ValueError(f"필수 필드가 누락되었습니다: {field}")
        
        # 선택적 필드 기본값 설정
        for field in template.optional_fields:
            if field not in kwargs:
                kwargs[field] = ""
        
        try:
            return template.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"템플릿 포맷팅 실패: {e}")
    
    def get_recommended_template(
        self, 
        data_quality_score: float, 
        token_limit: Optional[int] = None
    ) -> PromptType:
        """데이터 품질과 토큰 제한에 따른 추천 템플릿"""
        
        if token_limit and token_limit < 300:
            return PromptType.MINIMAL
        elif token_limit and token_limit < 600:
            return PromptType.BASIC
        elif data_quality_score >= 0.8:
            return PromptType.HIGH_QUALITY
        elif data_quality_score >= 0.5:
            return PromptType.MEDIUM_QUALITY
        else:
            return PromptType.BASIC
    
    def estimate_tokens(self, template_type: PromptType, **kwargs) -> int:
        """프롬프트 토큰 수 추정"""
        prompt = self.format_prompt(template_type, **kwargs)
        # 대략적인 토큰 수 계산 (영어: 1토큰=4자, 한국어: 1토큰=2자)
        korean_chars = sum(1 for c in prompt if ord(c) > 127)
        english_chars = len(prompt) - korean_chars
        estimated_tokens = (korean_chars // 2) + (english_chars // 4)
        return estimated_tokens


# 전역 템플릿 매니저 인스턴스
template_manager = PromptTemplateManager()


def get_optimized_prompt_with_templates(
    template_type: PromptType,
    script_text: str,
    learner_text: str,
    native_text: str,
    learner_timing: str = "",
    native_timing: str = "",
    prosody_context: str = "",
    rag_context: str = "",
) -> str:
    """템플릿을 사용한 최적화된 프롬프트 생성"""
    
    return template_manager.format_prompt(
        template_type,
        script_text=script_text,
        learner_text=learner_text,
        native_text=native_text,
        learner_timing=learner_timing,
        native_timing=native_timing,
        prosody_context=prosody_context,
        rag_context=rag_context,
    )


# 편의 함수들
def get_high_quality_prompt(**kwargs) -> str:
    """고품질 프롬프트 생성"""
    return get_optimized_prompt_with_templates(PromptType.HIGH_QUALITY, **kwargs)


def get_quick_prompt(**kwargs) -> str:
    """빠른 분석용 프롬프트 생성"""
    return get_optimized_prompt_with_templates(PromptType.BASIC, **kwargs)


def get_minimal_prompt(**kwargs) -> str:
    """최소 프롬프트 생성"""
    return get_optimized_prompt_with_templates(PromptType.MINIMAL, **kwargs) 