from promptings.CoT import CoTStrategy
from promptings.Direct import DirectStrategy
from promptings.Analogical import AnalogicalStrategy
from promptings.SelfPlanning import SelfPlanningStrategy
from promptings.Mapcoder_v1 import MapCoder as MapCoder
from promptings.MapCoder_v1_merge_kb import MapCoder as MpCoder_mergekb


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
        elif prompting_name == "MapCoder_mergekb":
            return MpCoder_mergekb

        else:
            raise Exception(f"Unknown prompting name {prompting_name}")
