import sys
import time
import traceback
from datetime import datetime

import sys
from datetime import datetime
from constants.paths import *

from models.Gemini import Gemini
from models.OpenAI import OpenAIModel

from results.Results import Results

from promptings.PromptingFactory import PromptingFactory
from datasets.DatasetFactory import DatasetFactory
from models.ModelFactory import ModelFactory
# 初始化基础配置
BASE_CONFIG = {
    "DATASET": "CC",
    "STRATEGY": [ "MapCoder_dfs15"] ,#  "MapCoder_dfs"],
    "MODEL_NAME": "ChatGPT", 
    "TEMPERATURE": 0,
    "PASS_AT_K": 1,
    "LANGUAGE": "Python3",
    "START_COUNT": 1,  # 新增起始计数
    "MAX_COUNT": 1,
    "task_amount": -1 # 新增最大计数
}



def run_single_count(current_count, cur_strategy):
    """运行单个计数任务"""
    config = BASE_CONFIG.copy()
    config["COUNT"] = str(current_count)
    config["STRATEGY"] = str(cur_strategy)
    name = "MapCoder_dfs15"
    run_name = f"{config['MODEL_NAME']}-{name}-{config['DATASET']}-{config['LANGUAGE']}-{config['TEMPERATURE']}-{config['PASS_AT_K']}-{config['COUNT']}"
    results_path = f"./outputs/{run_name}.jsonl"

    print(f"\n{''*3} 启动计数 #{current_count} {''*3}")
    print(f"运行标识: {run_name}")
    print(f"开始时间: {datetime.now()}\n{'='*60}")

    strategy = PromptingFactory.get_prompting_class(config['STRATEGY'])(
        model=ModelFactory.get_model_class(config['MODEL_NAME'])(temperature=config['TEMPERATURE']),
        data=DatasetFactory.get_dataset_class(config['DATASET'])(),
        language=config['LANGUAGE'],
        pass_at_k=config['PASS_AT_K'],
        results=Results(results_path),
        name=run_name,
        task_amount=config['task_amount']
    )

    max_attempts = 15  # 设置最多尝试次数
    attempt = 0

    while attempt < max_attempts:
        try:
            if strategy.run():
                print(f"\n{' ' * 3} 计数 #{current_count} 完成")
                return True
        except Exception as e:
            print(f"\n{' ' * 3} 错误: {str(e)}")
            print(f"错误详情:\n{traceback.format_exc()}")

        attempt += 1
        print(f"\n{' ' * 3} 5秒后重试计数 #{current_count}... (尝试 {attempt}/{max_attempts})")
        time.sleep(5)

def count_sequencer(strategy):
    """计数任务序列控制器"""
    current_count = BASE_CONFIG["START_COUNT"]
    
    while current_count <= BASE_CONFIG["MAX_COUNT"]:
        if run_single_count(current_count, strategy):
            current_count += 1  # 仅当成功完成时递增计数
        else:
            print(f"\n{''*3} 计数 #{current_count} 持续失败，保持当前计数")

    print(f"\n{''*3} 所有计数任务完成（1-{BASE_CONFIG['MAX_COUNT']}）")

if __name__ == "__main__":
    for element in BASE_CONFIG["STRATEGY"]:
        print(f"#########################\n自动递增计数器启动\n最大目标计数: {BASE_CONFIG['MAX_COUNT']}\n初始时间: {datetime.now()}\n##########################")

        count_sequencer(element)
    
        print(f"\n#########################\n最终结束时间: {datetime.now()}\n##########################") 