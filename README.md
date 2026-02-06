# Hymba+

Hymba+는 **모듈형 하이브리드 Transformer–SSM 아키텍처**를 구현하기 위한 실동작 스캐폴드입니다. 본 저장소는 과장된 약속을 피하고, 실행 가능한 핵심 구성과 확장 포인트를 제공하는 것을 목표로 합니다.

## 핵심 특징

- 모듈형 컴포넌트(Attention, SSM, MLP, Fusion, Norm, Embedding).
- 하이브리드 블록(Transformer / Mamba 유사 / Hybrid).
- 중첩 YAML → dataclass 로더로 구성 재현성 보장.
- Pretrain/SFT/RL/평가/내보내기 스크립트 제공.
- 검증용 노트북(중간 출력, 파라미터 수, 시각화).
- CUDA + Triton 사용 가능 시 SSM 게이팅 경로의 Triton 커널 사용.
- PyTorch SDP 경로를 통한 Flash Attention 사용(가능한 경우).
- GQA 지원 Attention 및 벡터화된 MoE.
- nanochat 스타일 학습 루프(코사인 LR, 워밍업, Grad Accum, BF16, Grad Clip).

## 빠른 시작

```bash
python scripts/evaluate.py --config configs/hymba_plus.yaml
```

## 노트북

- `notebooks/01_validation.ipynb`: 모델 초기화/전방 패스/파라미터/로그잇 시각화
- `notebooks/02_training_stages.ipynb`: 학습 단계별 흐름 이해 및 손실 곡선 시각화
- `notebooks/03_model_comparison.ipynb`: 모델 변형 비교와 지표 정리

## 상태

이 저장소는 **실행 가능한 스캐폴드**이며, 성능 최적화(FA3/Triton/FP8)는 향후 통합을 전제로 합니다. 상세 비판 리뷰와 로드맵은 `ARCHITECTURE.md`를 참고하세요.
