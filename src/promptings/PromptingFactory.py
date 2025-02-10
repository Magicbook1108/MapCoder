from promptings.CoT import CoTStrategy
from promptings.Direct import DirectStrategy
from promptings.Analogical import AnalogicalStrategy
from promptings.SelfPlanning import SelfPlanningStrategy
from promptings.Mapcoder_v1 import MapCoder
from promptings.MapCoder_merge_planning_1 import MapCoder as MapCoder_merge_1
from promptings.MapCoder_withoutKB_1 import MapCoder as MapCoder_withoutKB_1
from promptings.MapCoder_withoutKB_2 import MapCoder as MapCoder_withoutKB_2
from promptings.MapCoder_withoutKB_3 import MapCoder as MapCoder_withoutKB_3
from promptings.MapCoder_without_kb_debug import MapCoder as MapCoder_without_kb_debug
from promptings.MapCoder_dfs import MapCoder as MapCoder_dfs
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
        elif prompting_name =="MapCoder_single_merge":
            return MapCoder_merge_1
        elif prompting_name =="MapCoder_withoutKB_1":
            return MapCoder_withoutKB_1
        elif prompting_name =="MapCoder_withoutKB_2":
            return MapCoder_withoutKB_2
        elif prompting_name =="MapCoder_without_kb_debug":
            return MapCoder_without_kb_debug
        elif prompting_name =="MapCoder_withoutKB_3":
            return MapCoder_withoutKB_3
        elif prompting_name =="MapCoder_dfs":
            return MapCoder_dfs
        else:
            raise Exception(f"Unknown prompting name {prompting_name}")
