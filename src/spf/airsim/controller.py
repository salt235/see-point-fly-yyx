import time
import airsim
import threading
import queue
from collections import deque
from .drone_space import AirSimDroneActionSpace
from .action_projector import AirSimActionProjector
from ..base.drone_space import ActionPoint


class AirSimController:
    def __init__(
        self, adaptive_mode=True, image_width=1920, image_height=1080, config=None
    ):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.hover_settle_time = 0.8
        self.takeoff_settle_time = 1.5

        # Apply wind settings from config
        if config:
            wind_x = config.get("wind_x", 0.0)
            wind_y = config.get("wind_y", 0.0)
            wind_z = config.get("wind_z", 0.0)
            self.hover_settle_time = config.get("hover_settle_time", 0.8)
            self.takeoff_settle_time = config.get("takeoff_settle_time", 1.5)
            wind = airsim.Vector3r(wind_x, wind_y, wind_z)
            self.client.simSetWind(wind)
            print(f"Wind set to: X={wind_x}, Y={wind_y}, Z={wind_z} m/s (NED frame)")

        self.action_queue = queue.Queue()
        self.running = True
        self.action_history = deque(maxlen=5)
        self.adaptive_mode = adaptive_mode

        self.image_width = image_width
        self.image_height = image_height

        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

        self.action_space = AirSimDroneActionSpace()
        self.action_projector = AirSimActionProjector(
            image_width=self.image_width,
            image_height=self.image_height,
            adaptive_mode=self.adaptive_mode,
            config_path="config_airsim.yaml",
        )

        print(
            f"AirSimController initialized in {'adaptive' if self.adaptive_mode else 'obstacle'} mode."
        )

    def _control_loop(self):
        """Separate thread for drone control"""
        # 用一个独立的线程来执行命令，确保主线程不会被阻塞
        while self.running:
            try:
                action = self.action_queue.get(timeout=0.1)
                if action:
                    self._execute_action(action) # 执行单个命令
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Control error: {e}")

    def _execute_action(self, command_tuple):
        """Execute a single AirSim command"""
        # 命令是一个二元组
        command_type, params = command_tuple

        # 目前只处理旋转和移动命令
        try:
            if command_type == "rotate_yaw":
                angle = params["angle"] # 这个键必须存在
                yaw_rate = params.get("yaw_rate", 30.0) # 这个键可选，默认30度/秒
                duration = abs(angle) / yaw_rate # 根据角度和角速度算持续时间
                rate = yaw_rate if angle > 0 else -yaw_rate # 根据正负决定旋转方向
                print(f"Executing rotate_yaw: {angle:.1f}° at {yaw_rate:.1f}°/s")

                # 调用AirSim API执行旋转命令
                self.client.rotateByYawRateAsync(rate, duration).join()
                self.client.hoverAsync().join() # 旋转后立即进入悬停模式，确保稳定
                time.sleep(self.hover_settle_time)

            elif command_type == "move_velocity_body":
                vx = params["vx"]
                vy = params["vy"]
                vz = params["vz"]
                duration = params["duration"]
                yaw_rate = params.get("yaw_rate", 0)

                print(
                    f"[MOVEMENT] Executing move (body): vx={vx:.2f} m/s, vy={vy:.2f} m/s, vz={vz:.2f} m/s (DOWN+/UP-), yaw_rate={yaw_rate:.1f}°/s, duration={duration:.2f}s"
                )

                # Execute movement command
                # 配置偏航
                yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)

                # 以无人机自身为参考系移动，持续指定时间
                self.client.moveByVelocityBodyFrameAsync(
                    vx,
                    vy,
                    vz,
                    duration,
                    airsim.DrivetrainType.MaxDegreeOfFreedom, # 不强制无人机一定朝着运动方向飞，允许更自由的多方向运动
                    yaw_mode,
                ).join()

                # Immediately engage hover mode
                self.client.hoverAsync().join()
                time.sleep(self.hover_settle_time)

        except Exception as e:
            print(f"Command execution failed: {e}")
            import traceback

            traceback.print_exc()

    def wait_for_queue_empty(self, timeout=30, debug=False):
        """Wait until action queue is empty or timeout occurs"""
        # 判断还没有执行完的命令，而不是判断无人机是否完全执行完所有动作
        start_time = time.time()
        # 如果debug模式开启，打印初始队列大小
        if debug:
            print(f"Queue size before waiting: {self.action_queue.qsize()}")

        while not self.action_queue.empty():
            if debug:
                print(f"Queue not empty, remaining items: {self.action_queue.qsize()}")

            # 计算剩余时间，如果超时则返回False
            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                print("Warning: Timed out waiting for action queue to empty")
                return False
            time.sleep(0.1)

        if debug:
            print(f"Queue emptied after {time.time() - start_time:.2f} seconds")
        return True

    # 这个函数实际上是把动作放进队列，名字取得有点误导
    def execute_action(self, command_tuple):
        """Add action to queue"""
        self.action_queue.put(command_tuple)

    # 无人机安全收尾
    def stop(self):
        """Stop the control thread and reset drone"""
        self.running = False # 通知控制线程停止
        # 如果线程还在运行，等待它结束
        if self.control_thread.is_alive():
            self.control_thread.join()

        # 悬停降落锁电机，释放api控制权
        self.client.hoverAsync().join()
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

    def takeoff(self):
        """Takeoff the drone"""
        print("Taking off...")
        self.client.takeoffAsync().join() # 起飞api
        time.sleep(self.takeoff_settle_time)
        print("Takeoff complete")

    # 这个函数是核心，负责处理从自然语言指令到空间动作的转换，并执行第一个动作
    def process_spatial_command(self, current_frame, instruction: str):
        """Process command using spatial understanding system"""
        try:
            actions = self.action_projector.get_vlm_points(current_frame, instruction)

            if not actions:
                return "No valid actions identified" # 如果没有动作，直接返回提示信息

            response_text = "\n=== SINGLE ACTION MODE ===\n"

            action = actions[0] # 目前只执行第一个动作，后续可以扩展为执行多个动作或者根据某些策略选择动作
            if action is None:
                return "No valid action" 
            # 把本次要执行的空间动作打印出来
            response_text += (
                f"\n→ Moving: ({action.dx:.2f}, {action.dy:.2f}, {action.dz:.2f})"
            )
            self._execute_spatial_action(action, quiet=True) # 执行空间动作，quiet=True表示不在控制台打印执行细节，因为已经在上面打印了动作信息

            return response_text

        except Exception as e:
            print(f"Error: {e}")
            return "Error processing command"

    def _execute_spatial_action(self, action: ActionPoint, quiet: bool = False):
        """Execute a single spatial action"""
        commands = self.action_space.action_to_commands(action)

        for cmd_type, params in commands:
            if not quiet:
                print(f"Executing: {cmd_type} with {params}")
            self.execute_action((cmd_type, params))

'''
创建 AirSimController
↓
连接 AirSim，接管无人机
↓
启动后台控制线程
↓
takeoff() 起飞
↓
process_spatial_command(当前图像, 指令)
↓
action_projector.get_vlm_points(...) 生成空间动作
↓
取第一个 ActionPoint
↓
action_space.action_to_commands(action) 拆成底层命令
↓
execute_action(...) 把命令放入队列
↓
_control_loop() 后台线程从队列取命令
↓
_execute_action() 真正调用 AirSim API 执行
↓
每个动作执行后 hover 稳定
↓
stop() 悬停、降落、关电机、释放控制权
'''