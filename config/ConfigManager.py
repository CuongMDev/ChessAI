class ConfigManager:
    def __init__(self):
        self.mode = None
        self.NUM_SIMULATION = None
        self.EXPLORATION_WEIGHT = None
        self.TEMPERATURE_CUTOFF = None
        self.POLICY_SOFTMAX_TEMP = None
        self.FPU_VALUE = None

        # Các cấu hình cho từng chế độ
        self._settings = {
            "train": {
                "NUM_SIMULATION": 800,
                "EXPLORATION_WEIGHT": 1.32,
                "TEMPERATURE_CUTOFF": 60,
                'POLICY_SOFTMAX_TEMP': 1.45,
                "FPU_VALUE": 0.26
            },
            "eval": {
                "NUM_SIMULATION": 400,
                "EXPLORATION_WEIGHT": 1.32,
                "TEMPERATURE_CUTOFF": 0,
                'POLICY_SOFTMAX_TEMP': 1.4,
                "FPU_VALUE": 0.23
            },
            "play": {
                "NUM_SIMULATION": 200,
                "EXPLORATION_WEIGHT": 1.32,
                "TEMPERATURE_CUTOFF": 2,
                'POLICY_SOFTMAX_TEMP': 1.4,
                "FPU_VALUE": 0.23
            },
        }

    def set_mode(self, mode):
        if mode not in self._settings:
            raise ValueError(f"Unknown mode: {mode}")

        config = self._settings[mode]
        self.mode = mode
        self.NUM_SIMULATION = config["NUM_SIMULATION"]
        self.EXPLORATION_WEIGHT = config["EXPLORATION_WEIGHT"]
        self.TEMPERATURE_CUTOFF = config["TEMPERATURE_CUTOFF"]
        self.POLICY_SOFTMAX_TEMP = config["POLICY_SOFTMAX_TEMP"]
        self.FPU_VALUE = config["FPU_VALUE"]
