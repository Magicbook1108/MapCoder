from promptings.CoT import CoTStrategy
from promptings.Direct import DirectStrategy
from promptings.Analogical import AnalogicalStrategy
from promptings.SelfPlanning import SelfPlanningStrategy
from promptings.Mapcoder_v1 import MapCoder
from promptings.MapCoder_dfs import MapCoder as MapCoder_dfs

from promptings.MapCoder_debug import MapCoder as MapCoder_debug
from promptings.MapCoder_plan import MapCoder as MapCoder_plan
from promptings.MapCoder_retrieval import MapCoder as MapCoder_retrieval

from promptings.MapCoder_plan_debug import MapCoder as MapCoder_plan_debug
from promptings.MapCoder_retrieval_debug import MapCoder as MapCoder_retireval_debug
from promptings.MapCoder_retrieval_plan import MapCoder as MapCoder_retrieval_plan

class PromptingFactory:
    @staticmethod
    def get_prompting_class(prompting_name):
        if prompting_name == "CoT":
            return CoTStrategy
        elif prompting_name == "MapCoder":
            return MapCoder
        elif prompting_name == "Direct":
            return DirectStrategy
        elif prompting_name == "Analogical":
            return AnalogicalStrategy
        elif prompting_name == "SelfPlanning":
            return SelfPlanningStrategy
        elif prompting_name =="MapCoder_plan":
            return MapCoder_plan
        elif prompting_name =="MapCoder_dfs":
            return MapCoder_dfs
        elif prompting_name =="MapCoder_debug":
            return MapCoder_debug
        elif prompting_name =="MapCoder_plan":
            return MapCoder_plan
        elif prompting_name =="MapCoder_retrieval":
            return MapCoder_retrieval
        elif prompting_name =="MapCoder_retrieval_plan":
            return MapCoder_retrieval_plan
        elif prompting_name =="MapCoder_retrieval_debug":
            return MapCoder_retireval_debug
        elif prompting_name =="MapCoder_plan_debug":
            return MapCoder_plan_debug
        else:
            raise Exception(f"Unknown prompting name {prompting_name}")
