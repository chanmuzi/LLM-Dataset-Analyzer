#!/bin/bash

# 변수 설정
BENCHMARK_NAME=$1  # 분석할 데이터셋 이름
MODEL_NAME="gpt-4o-mini"  # 사용할 GPT 모델 이름
DOTENV_PATH="./.env"  # .env 파일 경로
SUBTASK_PROMPT_PATH="./prompts/subtask_prompt.txt"  # 하위 작업 프롬프트 파일 경로
SUMMARIZE_PROMPT_PATH="./prompts/summarize_prompt.txt"  # 요약 프롬프트 파일 경로
OUTPUT_DIR="./outputs"  # 결과 저장 디렉토리
SHOT_NUMBER=5  # 각 하위 작업에서 분석할 샘플 수

# BENCHMARK_NAME이 제공되지 않았을 경우 에러 메시지 출력 후 종료
if [ -z "$BENCHMARK_NAME" ]; then
  echo "Error: BENCHMARK_NAME is required. Usage: ./analyze.sh <benchmark_name>"
  exit 1
fi

# analyze.py 실행
python analyze.py \
  --benchmark_name "$BENCHMARK_NAME" \
  --model_name "$MODEL_NAME" \
  --dotenv_path "$DOTENV_PATH" \
  --subtask_prompt_path "$SUBTASK_PROMPT_PATH" \
  --summarize_prompt_path "$SUMMARIZE_PROMPT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --shot_number "$SHOT_NUMBER"