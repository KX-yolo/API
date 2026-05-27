from .singleTask import APRIL


class ATIO:
    def __init__(self):
        self.TRAIN_MAP = {
            "april": APRIL,
        }

    def getTrain(self, args):
        return self.TRAIN_MAP[args["model_name"]](args)
