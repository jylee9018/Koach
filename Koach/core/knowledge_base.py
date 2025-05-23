import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from config.settings import CURRENT_CONFIG, KNOWLEDGE_DIR

logger = logging.getLogger("Koach")


class KnowledgeBase:
    """한국어 발음 지식 베이스"""

    def __init__(
        self,
        knowledge_dir: str = "knowledge",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Args:
            knowledge_dir: 지식 베이스 디렉토리
            embedding_model: 임베딩 모델명
        """
        self.knowledge_dir = knowledge_dir
        os.makedirs(knowledge_dir, exist_ok=True)

        # 문서 저장소
        self.documents = []
        self.document_ids = []

        # 임베딩 모델 로드
        self.model = SentenceTransformer(embedding_model)

        # FAISS 인덱스 초기화
        self.index = None

        # 기본 지식 로드
        self.load_knowledge()

    def load_knowledge(self):
        """지식 베이스 로드"""
        try:
            # 지식 파일 목록
            knowledge_files = [
                "korean_pronunciation_rules.md",
                "common_pronunciation_errors.md",
                "pronunciation_tips.md",
            ]

            # 각 파일 로드
            for filename in knowledge_files:
                file_path = os.path.join(self.knowledge_dir, filename)
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        self.add_document(content)

            # FAISS 인덱스 생성
            if self.documents:
                self._create_index()

        except Exception as e:
            logger.error(f"지식 로드 중 오류 발생: {e}")

    def add_document(self, content: str):
        """문서 추가"""
        # 문서 ID 생성
        doc_id = str(len(self.documents))

        # 문서 저장
        self.documents.append({"id": doc_id, "content": content})
        self.document_ids.append(doc_id)

    def _create_index(self):
        """FAISS 인덱스 생성"""
        # 문서 임베딩
        embeddings = self.model.encode([doc["content"] for doc in self.documents])

        # FAISS 인덱스 생성
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype("float32"))

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """관련 지식 검색"""
        if not self.index or not self.documents:
            return []

        # 쿼리 임베딩
        query_embedding = self.model.encode([query])[0].astype("float32")

        # 검색
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), min(top_k, len(self.documents))
        )

        # 결과 반환
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append(
                    {
                        "content": self.documents[idx]["content"],
                        "score": float(distances[0][i]),
                    }
                )

        return results
