from promptings.CoT import CoTStrategy
from promptings.Direct import DirectStrategy
from promptings.Analogical import AnalogicalStrategy
from promptings.SelfPlanning import SelfPlanningStrategy
from promptings.MapCoder import MapCoder as MapCoder

from promptings.MapCoder_v1 import MapCoder as MapCoder_test
from promptings.MapCoder_v2 import MapCoder as MapCoder_v2
from promptings.MapCoder_withoutKB_1 import MapCoder as MapCoder_without_KB
from promptings.MapCoder_moreDebug import MapCoder as MapCoder_moreDebug
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
        elif prompting_name == "MapCoder_test":
            return MapCoder_v2
        elif prompting_name == "MapCoder_without_KB":
            return MapCoder_without_KB
        elif prompting_name == "MapCoder_moreDebug":
            return MapCoder_moreDebug
        else:
            raise Exception(f"Unknown prompting name {prompting_name}")
