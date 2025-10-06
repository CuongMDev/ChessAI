import torch
import torch.nn.init as init

class DeepNormInit:
    def __init__(self, encoder_layers, dtype=torch.float32):
        self.encoder_layers = encoder_layers
        self.model_dtype = dtype

        # Hệ số DeepNorm
        self.alpha = torch.tensor((2.0 * encoder_layers) ** 0.25, dtype=dtype)
        self.beta  = torch.tensor((8.0 * encoder_layers) ** -0.25, dtype=dtype)

    def __call__(self, tensor):
        """Cho phép gọi trực tiếp đối tượng như một hàm: initializer(tensor)"""
        return self.xavier_scaled_init(tensor)

    def xavier_scaled_init(self, tensor):
        """Khởi tạo kiểu Xavier Normal có nhân thêm hệ số β."""
        if tensor is not None:
            fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
            std = (float(self.beta) * (2.0 / (fan_in + fan_out))) ** 0.5
            with torch.no_grad():
                return tensor.normal_(0.0, std)
