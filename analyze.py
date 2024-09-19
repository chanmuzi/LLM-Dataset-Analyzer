import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import argparse
from tqdm.auto import tqdm
from utils import get_response, load_dataset_sample, get_prompt


def analyze_subtasks(dataset: Dict[str, List[Dict[str, str]]], subtask_prompt: str, client: OpenAI, model_name: str = "gpt-4o-mini") -> str:
    result_subtasks = "## 태스크별 요약\n\n"
    template = """{system_prompt}\n\n다음은 현재 하위 작업에 대한 예시들입니다:\n\n{few_shot_txt}\n\n이 작업을 분석하고 간단한 정의를 제공해 주세요."""
    
    # 태스크 이름 기준으로 하나씩 뽑아서 분석
    for idx, key in tqdm(enumerate(dataset.keys(), 1), desc="Analyzing subtasks", unit="subtask"):
        result_subtasks += f"### Task {idx}: {key}\n\n"
        data = dataset[key] # ex) key: 'abstract_algebra'
        data_keys = data[0].keys() # ex) data_keys: ['question', 'subject', 'choices', ... ]

        # few_shot_txt 생성
        few_shot_txt = ""
        for num, sample in enumerate(data, 1):
            few_shot_txt += f"## Example {num}\n"

            for key in data_keys:
                few_shot_txt += f"{key}: {sample[key]}\n"
            few_shot_txt += "\n\n"

            if num == 1:
                result_subtasks += f'**[예시]**\n\n'
                for key in data_keys:
                    result_subtasks += f"- {key}: {sample[key]}\n"
                result_subtasks += "\n"
                result_subtasks += f'**[설명]**\n\n'

        prompt = template.format(system_prompt=subtask_prompt, few_shot_txt=few_shot_txt)
        response = get_response(prompt, client, model_name=model_name)

        result_subtasks += response + "\n\n" + "---" + "\n\n"

    return result_subtasks


def summarize_subtasks(result_subtasks: str, summarize_prompt: str, benchmark_name: str, client: OpenAI, model_name: str = "gpt-4o-mini") -> str:
    print("Summarizing subtasks...")
    result_summary = f"## {benchmark_name}\n\n"
    template = "{summarize_prompt}\n\n다음은 분석한 벤치마크, **{benchmark_name}**의 결과입니다:\n\n{report}\n\n이 벤치마크의 전반적인 구조와 목적을 요약해 주세요."

    prompt = template.format(summarize_prompt=summarize_prompt, report=result_subtasks, benchmark_name=benchmark_name)
    response = get_response(prompt, client, model_name=model_name)

    result_summary += response + "\n\n" + "---" + "\n\n"
    result_summary += result_subtasks

    return result_summary


def main(args):
    # 환경변수 로드: OPENAI_API_KEY
    if args.dotenv_path:
        if os.path.exists(args.dotenv_path):
            load_dotenv(args.dotenv_path)
            print(f"Loaded .env file from {args.dotenv_path}")
        else:
            raise FileNotFoundError(f".env file not found at {args.dotenv_path}")
        
    if 'OPENAI_API_KEY' in os.environ:
        print("OPENAI_API_KEY is successfully loaded.")
    else:
        raise ValueError("OPENAI_API_KEY is not found in the loaded environment variables.")        

    # organization, project 설정 시 사용
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        # organization=os.getenv("OPENAI_ORGANIZATION"),
        # project=os.getenv("OPENAI_PROJECT")
    )

    dataset = load_dataset_sample(args.benchmark_name, args.shot_number)
    subtask_prompt, summarize_prompt = get_prompt(args.subtask_prompt_path, args.summarize_prompt_path)

    result_subtasks = analyze_subtasks(dataset, subtask_prompt, client, args.model_name)
    result_summary = summarize_subtasks(result_subtasks, summarize_prompt, args.benchmark_name, client, args.model_name)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/{args.benchmark_name.split('/')[0]}-{args.benchmark_name.split('/')[1]}.md", "w", encoding="utf-8") as f:
        f.write(result_summary)

    print(f"Result saved to {args.output_dir}/{args.benchmark_name}.md")
    print("Done!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--benchmark_name", "-b", type=str, required=True)
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4o-mini")
    parser.add_argument("--dotenv_path", "-d", type=str, default="./.env")
    parser.add_argument("--subtask_prompt_path", "-stp", type=str, default=None)
    parser.add_argument("--summarize_prompt_path", "-smp", type=str, default=None)
    parser.add_argument("--output_dir", "-o", type=str, default="./outputs")
    parser.add_argument("--shot_number", "-s", type=int, default=5)
    args = parser.parse_args()

    main(args)
