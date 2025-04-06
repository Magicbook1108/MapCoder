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

class configuration():
    def __init__(self, dataset, strategy, model_name = "GPT4", temperature = 0, language = "Python3", debugging = False, pass_at_k = 1):
        self.dataset = dataset
        self.strategy = strategy # list[str] e.g. ['MapCoder', 'CoT']
        self.model = model_name
        self.temperature = temperature
        self.language = language
        self.pass_at_k = pass_at_k
        self.trial = {"HumanEval" : 3,
                      "CC" : 3,
                      "MBPP":1,
                      "APPS":1,}
        self.max_trial = 10
        self.repeat_time = self.trial.get(self.dataset, 3)
        
    def set_trial(self,dataset,repeat_time):
        self.trial[dataset] = repeat_time    
    
    
    def run(self):
        while len(self.strategy) > 0:
            count = 1
            print(f"#########################\n自动递增计数器启动\n最大目标计数: {self.repeat_time}\n初始时间: {datetime.now()}\n##########################")
            
            strat = self.strategy.pop(0)
            
            while count <= self.repeat_time:                
                if self.run_single_count(strat, count):
                    count += 1
                else:
                    print(f"\n{''*3} 计数 #{count} 持续失败，保持当前计数")
                    
            print(f"\n#########################\n最终结束时间: {datetime.now()}\n##########################") 
        return True
        
    def run_single_count(self, strategy, count):
        run_name = f"{self.model}-{strategy}-{self.dataset}-{self.language}-{self.temperature}-{self.pass_at_k}-{count}"
        results_path = f"./outputs/{run_name}.jsonl"
        
        print(f"\n{''*3} 启动计数 #{count} {''*3}")
        print(f"运行标识: {run_name}")
        print(f"开始时间: {datetime.now()}\n{'='*60}")
        
        strategy = PromptingFactory.get_prompting_class(strategy)(
        model=ModelFactory.get_model_class(self.model)(self.temperature),
        data=DatasetFactory.get_dataset_class(self.dataset)(),
        language=self.language,
        pass_at_k=self.pass_at_k,
        results=Results(results_path),
        name=run_name,
        )
        
        max_retry = 10
        retry_count = 0
        
        while retry_count < max_retry:
            try:
                if strategy.run(self.dataset):
                    print(f"\n{' ' * 3} 计数 #{count} 完成")
                    return True
            except Exception as e:
                print(f"\n{' ' * 3} 错误: {str(e)}")
                print(f"错误详情:\n{traceback.format_exc()}")
                
            retry_count += 1
            for i in range(10, 0, -1):
                sys.stdout.write(f"\r重启倒计时: {i} 秒")
                sys.stdout.flush()
                time.sleep(1)

if __name__ == "__main__":

    human_config = configuration("HumanEval",[
        "Direct"])
    

    human_config.run()

    human_config = configuration("CC",[
                                         "MapCoder_plan",
                                         "MapCoder_debug",
                                         "MapCoder_retrieval",
                                         "MapCoder_retrieval_debug",
                                         "MapCoder_plan_debug"])
    

    human_config.run()