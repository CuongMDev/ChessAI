# Chess Game Based on LC0

## 🔧 Training

1. Modify the configuration files in the **config** folder.
2. Download Syzygy endgame tablebases and place all files into the **Syzygy** folder for improved training performance.
3. To start fresh training:
   - Remove the **checkpoint.pth** file in the **saved_model** folder.
4. To resume training:
   - Keep the existing **checkpoint.pth** file.
5. Run **mcuong-train.py** in the **Train** folder to start training.

## 🧠 Getting the AI Process

1. If you want to use the **CUDA version**:
   - Download all CUDA libraries listed in **cuda_lib/lib.txt**.
   - Place them in the **cuda_lib** folder.
2. Run **BuildONNX.py** to export the model to ONNX format.
3. Run **BuildProcess.py** using the dependencies listed in **build_process_requirements.txt** to build the AI process.
