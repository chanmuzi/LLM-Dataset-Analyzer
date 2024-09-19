# LLM Dataset Analyzer 🔍

LLM Dataset Analyzer는 대규모 언어 모델(LLM) 관련 데이터셋의 특징을 쉽고 빠르게 분석할 수 있는 도구입니다. 🚀

## 주요 기능 ✨

- 🤗 Hugging Face의 datasets 라이브러리를 통해 데이터셋 로드
- 📊 데이터셋의 하위 작업(subtask) 별 샘플 추출 및 분석
- 📝 GPT 모델을 활용한 데이터셋 특징 요약
- 📄 마크다운 형식의 분석 보고서 생성

## 시작하기 🚀

### 1. 설치
```bash
git clone https://github.com/your-username/LLM-Dataset-Analyzer.git
cd LLM-Dataset-Analyzer
pip install -r requirements.txt
```

### 2. 환경 설정

`.env` 파일을 생성하고 OpenAI API 키를 추가합니다:  
`OPENAI_API_KEY=your_api_key_here`


### 3. 실행

다음 명령어로 분석을 시작합니다:
```bash
python analyze.py --benchmark_name huggingface/dataset_name
```
또는
```bash
bash analyze.sh hails/mmlu_no_train
```


## 주요 매개변수 🛠️

- `--benchmark_name` or `-b`: 분석할 데이터셋 이름 (필수)
- `--model_name` or `-m`: 사용할 GPT 모델 이름 (기본값: "gpt-4o-mini")
- `--shot_number` or `-s`: 각 하위 작업에서 분석할 샘플 수 (기본값: 5)
- `--output_dir` or `-o`: 결과 저장 디렉토리 (기본값: "./output")

## 주의사항 ⚠️

- 이 도구는 Hugging Face의 datasets 라이브러리를 통해 다운로드 가능한 데이터셋 형식만 지원합니다.
- 분석에는 OpenAI API를 사용하므로, 사용량에 따라 비용이 발생할 수 있습니다.