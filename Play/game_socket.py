import io
import json
import socket
import sys
import traceback
from threading import Thread, Event

from Play.GamePlay import GamePlay
from config.config import host, port

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # Chỉnh output thành UTF-8

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((host, port))
server.listen(1)
print("Server Python Ready", flush=True)

conn, addr = server.accept()
print(f"Kết nối từ: {addr}")

need_cancel = Event()
gameplay = GamePlay()

def send_ai_act(move_uci=None):
    """
    :param move_uci: move_uci=None => Just send result
    :return:
    """
    # Gửi kết quả trở lại conn dưới dạng JSON
    data = {"move_uci": move_uci,
            "can_draw": gameplay.can_claim_draw(),
            "result": gameplay.result()}
    json_data = json.dumps(data) + "\n"
    conn.sendall(json_data.encode())

ai_thread = None

def ai_move_thread():
    move_uci, done = gameplay.ai_play()
    if not need_cancel.is_set():
        send_ai_act(move_uci)

buffer = ""
while True:
    chunk = conn.recv(1024).decode()
    if not chunk:
        break
    buffer += chunk

    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        if not line.strip():
            continue

        try:
            received_data = json.loads(line)

            if "move_uci" in received_data:
                move_uci = received_data["move_uci"]

                # Chơi
                done = gameplay.play(move_uci)
                need_cancel.clear()

                if not done:
                    ai_thread = Thread(target=ai_move_thread)
                    ai_thread.start()
                elif not need_cancel.is_set():
                    send_ai_act()

            elif "reset" in received_data and "human_play_first" in received_data and "fen" in received_data:
                need_cancel.set()
                if ai_thread is not None:
                    ai_thread.join() # wait ai

                human_play_first = received_data["human_play_first"]
                fen = received_data["fen"]
                gameplay.reset(fen)
                need_cancel.clear()
                if not human_play_first:
                    #ai_thread = Thread(target=ai_move_thread)
                    #ai_thread.start()
                    done = False
                    while not done:
                        move_uci, done = gameplay.ai_play()
                        send_ai_act(move_uci)

            elif "rollback" in received_data:
                need_cancel.set()
                if ai_thread is not None and ai_thread.is_alive():
                    ai_thread.join() # wait ai
                    gameplay.rollback() # ai moved, so need rollback

                gameplay.rollback()
                send_ai_act()

            elif "thinking_ability" in received_data:
                num_simulation = received_data["thinking_ability"] * 100
                gameplay.set_num_simulation(num_simulation)

        except Exception as e:
            # Lấy chuỗi traceback đầy đủ
            tb_str = traceback.format_exc()

            # Gửi lỗi dạng JSON (có thêm thông tin traceback nếu muốn)
            error_dict = {
                "error": str(e),
                "traceback": tb_str
            }
            conn.sendall(json.dumps(error_dict).encode())

            raise e

conn.close()
