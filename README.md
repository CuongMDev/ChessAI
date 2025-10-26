# Chess AI Based on MCTS (Monte Carlo Tree Search)

This repository contains the **AI engine** used by the [Chess GUI](https://github.com/CuongMDev/ChessGame).  
The AI is inspired by **AlphaZero** and uses **MCTS (Monte Carlo Tree Search)** combined with a **deep neural network** to evaluate board positions.

---

## 🔧 Training

1. Modify the configuration files in the **config** folder.
2. Download the following files:
   - **Syzygy endgame tablebases** → place all files into the **Syzygy** folder to improve training performance.
   - **Gaviota endgame tablebases** → place all files into the **Gaviota** folder to improve training performance.
   - **pretrain.pgn** → put into the **pretrain_data** folder if the `PRETRAIN` option is enabled.
   - **opening.pgn** → put into the **pretrain_data** folder for model testing.
3. To start **fresh training**:
   - Remove the `checkpoint.pth` file in the **saved_model** folder.
4. To **resume training**:
   - Keep the existing `checkpoint.pth` file.
5. Run `mcuong-train.py` in the **Train** folder to start training.

---

## 🧠 Building the AI Process

1. If you want to use the **CUDA version**:
   - Download all CUDA libraries listed in **cuda_lib/lib.txt**.
   - Place them in the **cuda_lib** folder.
2. Run `BuildONNX.py` to export the model to **ONNX** format.
3. Run `BuildProcess.py` using the dependencies listed in **build_process_requirements.txt** to build the AI process.

---

## 💡 Features

- Neural network trained via **self-play** (AlphaZero-style).
- Uses **MCTS** for decision making, not Minimax.
- Can export and run independently as a **process** communicating with the GUI via sockets or pipes.
- Supports both **CPU** and **CUDA** acceleration.

---

## 🧩 Related Repository

- [Chess GUI (JavaFX)](https://github.com/CuongMDev/ChessGame): A user interface where players can play against the AI, adjust difficulty, and manage game settings.

---

## 🏆 Author

**Mạnh Cường Nguyễn**  
📧 Contact: [GitHub Profile](https://github.com/CuongMDev)
