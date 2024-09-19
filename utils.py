from openai import OpenAI
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

def get_response(prompt: str, client: OpenAI, model_name: str = "gpt-4o-mini") -> str:
    # 모델 추론 결과를 반환하는 함수
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt}
        ],
        # response_format={ "type": "text" }
    )
    return response.choices[0].message.content


def get_first_available_split(benchmark_name: str, config: str) -> str:
    # 해당 서브 태스크의 데이터셋 분리 목록 추출. ex) 'train', 'validation', 'test'
    splits = get_dataset_split_names(benchmark_name, config)

    # 선호 목록이 있다면 선호 목록 순서대로 존재하는 데이터셋 분리 목록 추출
    preferred_splits = ['train', 'validation', 'val', 'test']
    for split in preferred_splits:
        if split in splits:
            return split
    
    # 선호 목록에 없는 경우 존재하는 데이터셋 분리 목록 중 첫 번째 목록 반환
    return splits[0] if splits else None


def load_dataset_sample(benchmark_name: str, shot_number: int = 5) -> Dict[str, List[Dict[str, str]]]:
    print(f"The dataset for {benchmark_name} will be loaded.")
    
    # 해당 데이터셋에 존재하는 데이터 목록 저장. ex) 'all', 'abstract_algebra' 등
    config_names = get_dataset_config_names(benchmark_name)
    config_names = [config for config in config_names if config != 'all']
    print(f"The number of subtasks: {len(config_names)}")
    print(f"The number of samples per subtask: {shot_number}")

    # 각 데이터 목록에 대해 {shot_number}개의 샘플만 로드
    datasets = {}
    for idx, config in enumerate(tqdm(config_names, desc="Loading subtasks", unit="subtask")):
        print(f"\rCurrent loading subtask: {config}", end="", flush=True)
        split = get_first_available_split(benchmark_name, config)
        if split:
            dataset = load_dataset(benchmark_name, config, split=split, streaming=True)
            datasets[config] = list(dataset.take(shot_number))
        else:
            print(f"Warning: No splits available for {benchmark_name}/{config}")

    print(f"The dataset for {benchmark_name} has been loaded.")
    return datasets


def get_prompt(subtask_prompt_path: str = None, summarize_prompt_path: str = None) -> Tuple[str, str]:
    # 하위 태스크별 프롬프트 로드
    subtask_prompt_path = subtask_prompt_path or "./prompts/subtask_prompt.txt"
    print(f"Loading the subtask prompt from {subtask_prompt_path}...")
    with open(subtask_prompt_path, "r") as file:
        subtask_prompt = file.readlines()

    subtask_prompt = "".join(subtask_prompt)

    # 전체 결과 요약 프롬프트 로드
    summarize_prompt_path = summarize_prompt_path or "./prompts/summarize_prompt.txt"
    print(f"Loading the summarize prompt from {summarize_prompt_path}...")
    with open(summarize_prompt_path, "r") as file:

        summarize_prompt = file.readlines()
    summarize_prompt = "".join(summarize_prompt)
    
    return subtask_prompt, summarize_prompt
