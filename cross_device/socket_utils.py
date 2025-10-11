import socket
import threading
import logging


class DualSocket:
    """
    双设备Socket通信管理类

    该类用于实现两台设备（如安卓手机）之间的双向通信，
    支持同时作为服务器监听端口和作为客户端发送消息，通过回调函数处理接收到的消息，
    适用于需要设备间实时指令交互和数据传输的场景（如多设备协同完成任务）。
    """

    def __init__(self, listen_port: int, peer_port: int, device_name: str = "设备"):
        self.listen_port = listen_port  # 监听端口
        self.peer_port = peer_port  # 对方设备端口
        self.device_name = device_name  # 设备名称
        self.is_running = False
        self.server_thread = None
        self.message_handler = None

    def start_server(self, handler):
        self.message_handler = handler
        self.is_running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()

    def _server_loop(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", self.listen_port))
            s.listen(1)
            while self.is_running:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024).decode(encoding="utf-8")
                    if data and self.message_handler:
                        response = self.message_handler(data)
                        conn.sendall(response.encode(encoding="utf-8"))

    def send_message(self, message: str) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("localhost", self.peer_port))
                s.sendall(message.encode(encoding="utf-8"))
                response = s.recv(1024).decode(encoding="utf-8")
                return response
        except Exception as e:
            return ""

    def stop(self):
        self.is_running = False
        if self.server_thread:
            self.server_thread.join(timeout=1)
