import os
import sys

import numpy as np
from numba import njit
import onnxruntime as ort

class Utils:
    @staticmethod
    @njit
    def compute_kld(p_policy, visit_counts, eps=1e-8):
        """
        p_policy: numpy array, xác suất từ mạng neural (đã qua softmax), shape (N,)
        visit_counts: numpy array, số lượt visit của MCTS, shape (N,)
        eps: để tránh log(0)

        Trả về: giá trị KLD (scalar)
        """
        # Normalize visit counts để tạo π_target
        p_target = visit_counts / (np.sum(visit_counts) + eps)

        # Đảm bảo không có log(0)
        p_policy = np.clip(p_policy, eps, 1.0)
        p_target = np.clip(p_target, eps, 1.0)

        # Tính KL divergence: sum p_target * log(p_target / p_policy)
        kld = np.sum(p_target * np.log(p_target / p_policy))
        return kld

    @staticmethod
    def resource_path(relative_path):
        """Lấy đúng đường dẫn resource trong PyInstaller"""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    @staticmethod
    @njit
    def flip_rows(bb: np.int64) -> np.int64:
        x = np.uint64(bb)

        x = ((x & 0x00000000000000FF) << 56) | ((x & 0xFF00000000000000) >> 56) | \
            ((x & 0x000000000000FF00) << 40) | ((x & 0x00FF000000000000) >> 40) | \
            ((x & 0x0000000000FF0000) << 24) | ((x & 0x0000FF0000000000) >> 24) | \
            ((x & 0x00000000FF000000) << 8) | ((x & 0x000000FF00000000) >> 8)

        return np.int64(x)

    @staticmethod
    def get_session(model_path):
        try:
            session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
            print("✅ Sử dụng GPU (CUDA)")
        except Exception as e:
            print("⚠️ Không có GPU, fallback sang CPU")
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        return session

