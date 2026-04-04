import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import pygame
import sys
import time
import os
import json
import random

# =========================
# Configuration / Constants
# =========================

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
FPS = 30

HP_MAX = 3
ICON_SIZE = 80
ICON_LIFETIME = 5.0       # seconds before an icon times out (causes damage)
SPAWN_INTERVAL = 3.0      # seconds between spawn attempts
MAX_ICONS = 4             # up to 4 icons (one per quadrant conceptually)

LEADERBOARD_FILE = "leaderboard.json"
SNAPSHOT_DIR = "snapshots"

# =========================
# Gesture Detection Module
# =========================

class GestureDetector:
    """
    使用 MediaPipe Tasks HandLandmarker：
    - 获取手 21 个关键点
    - 获取 handedness (Left / Right)
    - 基于关键点简单判定 rock / paper / scissors
    """

    def __init__(self, model_path="hand_landmarker.task"):
        # 初始化 Hand Landmarker
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=mp_vision.RunningMode.VIDEO,  # 必须设为 VIDEO 才能使用 detect_for_video
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

        # 用于 detect_for_video 的时间戳（毫秒）
        self._timestamp_ms = 0

        # 连续3帧一样的手势才算稳定，避免误判
        self.history = {"Left": [], "Right": []}
        self.finger_count_history = []

    def count_fingers(self, lm_list):
        """
        计算伸出的手指数量（包括拇指）
        返回: 0-5
        """
        def is_finger_open(tip_id, pip_id):
            return lm_list[tip_id].y < lm_list[pip_id].y
        
        def is_thumb_open():
            # 拇指判断：比较x坐标（横向伸展）
            # 指尖比IP更靠外则认为伸直
            return lm_list[4].x < lm_list[3].x
        
        thumb_open = is_thumb_open()
        index_open = is_finger_open(8, 6)
        middle_open = is_finger_open(12, 10)
        ring_open = is_finger_open(16, 14)
        pinky_open = is_finger_open(20, 18)
        
        return sum([thumb_open, index_open, middle_open, ring_open, pinky_open])

    def detect_finger_count(self, frame_rgb, image_width, image_height):
        """
        检测一只手的手指数量（用于数学游戏）
        返回: {'finger_count': int, 'hand_up_flag': bool, 'stable': bool}
        - hand_up_flag: True=检测到手, False=没有检测到手
        - finger_count: 0-5 (握拳为0，只有hand_up_flag=True时有效)
        - stable: True=连续4帧相同，答案已稳定
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += int(1000 / 30)
        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)
        
        if not result.hand_landmarks:
            self.finger_count_history = []
            return {"finger_count": 0, "hand_up_flag": False, "stable": False}
        
        # 只检测第一只手
        lm_list = result.hand_landmarks[0]
        raw_count = self.count_fingers(lm_list)
        
        # 添加到历史记录
        self.finger_count_history.append(raw_count)
        if len(self.finger_count_history) > 4:
            self.finger_count_history.pop(0)
        
        # 检查是否连续4帧相同
        stable = False
        confirmed_count = 0
        if len(self.finger_count_history) == 4:
            if (self.finger_count_history[0] == 
                self.finger_count_history[1] == 
                self.finger_count_history[2] == 
                self.finger_count_history[3]):
                confirmed_count = self.finger_count_history[0]
                stable = True
        
        return {
            "finger_count": confirmed_count, 
            "hand_up_flag": True, 
            "stable": stable,
            "current_count": raw_count  # 当前帧检测到的数量（用于显示）
        }

    def detect(self, frame_rgb, image_width, image_height):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += int(1000 / 30)
        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)
        hands_info = []

        if not result.hand_landmarks:
            self.history = {"Left": [], "Right": []}
            return hands_info

        for hand_idx, lm_list in enumerate(result.hand_landmarks):
            handedness_list = result.handedness[hand_idx]
            original_label = handedness_list[0].category_name
            handed_label = "Right" if original_label == "Left" else "Left"

            # 1. 获取原始手势与中心坐标
            raw_gesture, center = self._classify_hand(lm_list, image_width, image_height)
            
            # 2. 维护长度为 3 的滑动窗口
            self.history[handed_label].append(raw_gesture)
            if len(self.history[handed_label]) > 3:
                self.history[handed_label].pop(0)

            # 3. 只有连续 3 帧相同才返回该手势
            confirmed_gesture = "none"
            quadrant = "None"
            if len(self.history[handed_label]) == 3:
                if (self.history[handed_label][0] == self.history[handed_label][1] == self.history[handed_label][2] and self.history[handed_label][0] != "none"):
                    confirmed_gesture = self.history[handed_label][0]
                    
                    # --- 整合：计算象限并统一打印 ---
                    cx, cy = center
                    is_left = cx < image_width // 2
                    is_top = cy < image_height // 2
                    quadrant = f"{'Top' if is_top else 'Bottom'}-{'Left' if is_left else 'Right'}"
                    
                    print(f"[DEBUG] Confirmed: Hand={handed_label} \vert  Gesture={confirmed_gesture} \vert  Quadrant={quadrant}")

            # 将结果加入列表，即使是 none 也可以加入，如果只想获取有效手势，可以加个 if confirmed_gesture != "none"
            hands_info.append({
                "gesture": confirmed_gesture,
                "handedness": handed_label,
                "center": center,
                "quadrant": quadrant if confirmed_gesture != "none" else "None" # 也可以把象限传出去给游戏逻辑用
            })
            
        return hands_info



    def _classify_hand(self, lm_list, width, height):
        """
        lm_list: 长度 21 的关键点列表，每个元素有 .x, .y (0~1 归一化)

        返回:
        - gesture: "rock" / "paper" / "scissors" / "none"
        - center_px: (cx, cy) 像素坐标
        """
        # 手中心 = 所有关键点平均位置
        cx = sum(lm.x for lm in lm_list) / len(lm_list)
        cy = sum(lm.y for lm in lm_list) / len(lm_list)
        center_px = (int(cx * width), int(cy * height))

        # 定义几个常用关键点索引（和老的 solutions 一样）
        # Thumb: 4 (tip), 3 (IP) -- 这里暂时不用
        # Index: 8 (tip), 6 (PIP)
        # Middle: 12 (tip), 10 (PIP)
        # Ring: 16 (tip), 14 (PIP)
        # Pinky: 20 (tip), 18 (PIP)

        def is_finger_open(tip_id, pip_id):
            # y 越小越靠上；指尖比 PIP 更“上”则认为伸直
            return lm_list[tip_id].y < lm_list[pip_id].y

        index_open = is_finger_open(8, 6)
        middle_open = is_finger_open(12, 10)
        ring_open = is_finger_open(16, 14)
        pinky_open = is_finger_open(20, 18)

        open_count = sum(
            [index_open, middle_open, ring_open, pinky_open]
        )

        # 简单规则：
        # rock: 四指都弯曲
        # paper: 四指都伸直
        # scissors: 只伸食指 + 中指
        if open_count == 0:
            gesture = "rock"
        elif index_open and middle_open and (not ring_open) and (not pinky_open):
            gesture = "scissors"
        elif open_count == 4:
            gesture = "paper"
        else:
            gesture = "none"

        return gesture, center_px



# =========================
# Icon / Enemy Representation
# =========================

class Icon:
    """
    Represents an enemy icon in one of the four screen quadrants.
    Each icon has:
      - gesture: 'rock', 'paper', or 'scissors' (enemy gesture)
      - required_handedness: 'Left' or 'Right' (icon color encodes this)
      - rect: pygame.Rect defining the icon's hit-box
      - spawn_time: timestamp when it appeared
      - lifetime: seconds it stays on screen before timing out
    """

    def __init__(self, gesture, rect, required_handedness, spawn_time, lifetime, quadrant):
        self.gesture = gesture
        self.draw_rect = rect 
        # 现在的 hit_rect 实际上不再需要精细的矩形碰撞，但保留它用于绘制图标位置即可
        self.rect = rect 
        self.required_handedness = required_handedness
        self.quadrant = quadrant  # 新增属性：记录象限
        self.spawn_time = spawn_time
        self.lifetime = lifetime

        # 颜色逻辑保持不变
        self.color = (255, 0, 0) if self.required_handedness == "Left" else (0, 255, 0)
    
    @staticmethod
    def counter_gesture(gesture):
        """
        Returns the gesture that beats the enemy gesture.
          rock -> paper
          paper -> scissors
          scissors -> rock
        """
        if gesture == "rock":
            return "paper"
        elif gesture == "paper":
            return "scissors"
        elif gesture == "scissors":
            return "rock"
        return "none"


# =========================
# Main Game Class
# =========================

class HandGestureBattleGame:
    """
    Main game class integrating:
      - OpenCV camera input
      - MediaPipe gesture detection
      - Pygame game loop, UI, state management
    States:
      - START: wait for player to show SCISSORS to start
      - RUNNING: spawn icons, detect hits/misses
      - GAME_OVER: snapshot taken, leaderboard shown, wait for restart
    """

    def __init__(self):
        # Ensure snapshot directory exists
        if not os.path.exists(SNAPSHOT_DIR):
            os.makedirs(SNAPSHOT_DIR)

        pygame.init()
        pygame.display.set_caption("Hand Gesture Battle Game")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("arial", 24)
        self.big_font = pygame.font.SysFont("arial", 36, bold=True)

        # OpenCV camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

        # Gesture detector
        self.detector = GestureDetector()

        # Game state variables
        self.state = "START"
        self.hp = HP_MAX
        self.score = 0
        self.icons = []
        self.last_spawn_time = 0.0

        self.frame_count = 0 # 用于控制检测频率
        self.last_hands_info = []

        self.running = True

        #--- 新增：记录上一个生成的象限索引 ---
        self.last_quadrant = -1 
        
        # Snapshot / leaderboard
        self.snapshot_surface = None
        self.snapshot_file = None
        self.leaderboard = self.load_leaderboard()


        # 新增： 记录进入 READY 状态的时间
        self.state = "MODE_SELECT"
        self.game_mode = 1
        self.ready_start_time = 0
        self.mode_select_waiting_reset = False  # 模式选择的手势复位标志
        
        # Load gesture images
        self.gesture_images = {
            "rock": pygame.image.load("Game assets/rock.png"),
            "paper": pygame.image.load("Game assets/paper.png"),
            "scissors": pygame.image.load("Game assets/scissors.png")
        }
        for img in self.gesture_images.values():
            img.set_colorkey((255, 255, 255))
        
        # Math game variables
        self.math_question_num = 0
        self.math_total_questions = 10
        self.math_correct_count = 0
        self.math_question = ""
        self.math_answer = 0
        self.math_question_time = 0
        self.math_time_limit = 10.0
        self.math_finger_count = 0
        self.math_hand_up = False  # 是否检测到手
        self.math_waiting_reset = False  # 需要手势复位才能读下一个答案
        self.math_finger_stable = False  # 手指数量是否稳定（连续4帧相同）
        self.math_current_finger = 0  # 当前帧检测到的手指数
        
        # Fruit memory game variables
        self.fruit_game_mode = 1  # 1=普通模式, 2=挑战模式
        self.fruit_question_num = 0
        self.fruit_total_questions = 5  # 普通模式5题
        self.fruit_max_questions = 9  # 挑战模式上限为9题
        self.fruit_correct_count = 0
        self.fruit_consecutive_correct = 0  # 连续正确次数
        self.fruit_num_types = 2  # 当前水果种类数
        self.fruit_display_time = 5.0
        self.fruit_answer_time = 10.0
        self.fruit_types = ["Apple", "Banana"]
        self.fruit_positions = []
        self.fruit_counts = {}
        self.fruit_target = ""
        self.fruit_answer = 0
        self.fruit_question_time = 0
        self.fruit_finger_count = 0
        self.fruit_hand_up = False  # 是否检测到手
        self.fruit_waiting_reset = False  # 需要手势复位才能读下一个答案
        self.fruit_mode_select_time = 0  # 模式选择时间
        self.fruit_mode_waiting_reset = False  # 水果模式选择的手势复位标志
        self.mode_select_time = 0  # 模式选择开始时间
        
        # Load fruit images
        self.fruit_images = {
            "Apple": pygame.image.load("Game assets/Apple.png"),
            "Banana": pygame.image.load("Game assets/Banana.png"),
            "Grapes": pygame.image.load("Game assets/Grapes.png"),
            "Mango": pygame.image.load("Game assets/Mango.png"),
            "Pineapple": pygame.image.load("Game assets/Pineapple.png"),
            "Watermelon": pygame.image.load("Game assets/Watermelon.png")
        }
        for img in self.fruit_images.values():
            img.set_colorkey((255, 255, 255))


    # -------------
    # Leaderboard
    # -------------

    def load_leaderboard(self):
        if not os.path.exists(LEADERBOARD_FILE):
            return []
        try:
            with open(LEADERBOARD_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        return []

    def save_leaderboard(self):
        try:
            with open(LEADERBOARD_FILE, "w", encoding="utf-8") as f:
                json.dump(self.leaderboard, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_to_leaderboard(self, score, snapshot_file):
        entry = {
            "name": "Player",
            "score": score,
            "snapshot": snapshot_file
        }
        self.leaderboard.append(entry)
        # Sort descending by score, keep top 10
        self.leaderboard.sort(key=lambda e: e["score"], reverse=True)
        self.leaderboard = self.leaderboard[:10]
        self.save_leaderboard()

    # -------------
    # Icon Spawning
    # -------------

    
    def spawn_icon(self):
        """
        Spawn an icon in one of the four quadrants with smarter placement:

        1. 优先选择当前没有图标的象限（避免多个敌人挤在同一象限）。
        2. 如果有多个可选象限，则尽量不与上一次生成的象限相同。
        """

        # 随机敌人手势和需要的左右手
        gesture = random.choice(["rock", "paper", "scissors"])
        required_handedness = random.choice(["Left", "Right"])

        # 象限名称列表，与索引 0~3 对应
        quad_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
        quad_map = {name: idx for idx, name in enumerate(quad_names)}

        # 1. 统计当前屏幕上已有图标占用的象限索引
        active_quads = set()
        for icon in self.icons:
            idx = quad_map.get(icon.quadrant, None)
            if idx is not None:
                active_quads.add(idx)

        # 2. 先尝试在“未被占用的象限”中随机
        all_quads = [0, 1, 2, 3]
        available_quads = [q for q in all_quads if q not in active_quads]

        # 如果所有象限都被占满，则退化为所有象限都可用
        if not available_quads:
            available_quads = all_quads.copy()

        # 3. 在有多个可选象限时，尽量避免与上一次生成的象限相同
        if self.last_quadrant in available_quads and len(available_quads) > 1:
            available_quads = [q for q in available_quads if q != self.last_quadrant]

        # 4. 最终从候选象限中选择一个
        quad = random.choice(available_quads)
        self.last_quadrant = quad  # 记录这一次的象限

        quad_name = quad_names[quad]

        # 5. 根据象限计算图标中心位置
        if quad == 0:  # Top-Left
            center = (WINDOW_WIDTH // 4, WINDOW_HEIGHT // 4)
        elif quad == 1:  # Top-Right
            center = (3 * WINDOW_WIDTH // 4, WINDOW_HEIGHT // 4)
        elif quad == 2:  # Bottom-Left
            center = (WINDOW_WIDTH // 4, 3 * WINDOW_HEIGHT // 4)
        else:  # quad == 3, Bottom-Right
            center = (3 * WINDOW_WIDTH // 4, 3 * WINDOW_HEIGHT // 4)

        # 6. 创建 Pygame Rect 和 Icon 实例
        rect = pygame.Rect(0, 0, ICON_SIZE, ICON_SIZE)
        rect.center = center

        now = time.time()
        icon = Icon(
            gesture=gesture,
            rect=rect,
            required_handedness=required_handedness,
            spawn_time=now,
            lifetime=ICON_LIFETIME,
            quadrant=quad_name  # 记录象限名称
        )

        self.icons.append(icon)
        self.last_spawn_time = now


    # -------------
    # Game State Helpers
    # -------------

    def reset_game(self):
        self.state = "MODE_SELECT"
        self.game_mode = 1
        self.hp = HP_MAX
        self.score = 0
        self.icons = []
        self.last_spawn_time = 0.0
        self.snapshot_surface = None
        self.snapshot_file = None
        self.math_question_num = 0
        self.math_correct_count = 0
        self.math_question = ""
        self.math_answer = 0
        self.math_finger_count = 0
        self.math_hand_up = False
        self.math_waiting_reset = False
        self.mode_select_waiting_reset = False
        self.mode_select_time = 0
        self.fruit_game_mode = 1
        self.fruit_question_num = 0
        self.fruit_correct_count = 0
        self.fruit_consecutive_correct = 0
        self.fruit_num_types = 2
        self.fruit_positions = []
        self.fruit_counts = {}
        self.fruit_target = ""
        self.fruit_answer = 0
        self.fruit_finger_count = 0
        self.fruit_hand_up = False
        self.fruit_waiting_reset = False
        self.fruit_mode_waiting_reset = False

    def trigger_game_over(self, frame_bgr):
        """
        Called when HP reaches 0:
          - Take snapshot (save to disk)
          - Prepare snapshot surface
          - Add to leaderboard
          - Switch to GAME_OVER state
        """
        self.state = "GAME_OVER"

        timestamp = int(time.time())
        snapshot_filename = os.path.join(
            SNAPSHOT_DIR, f"snapshot_{timestamp}.png"
        )
        # Save snapshot (BGR -> file)
        cv2.imwrite(snapshot_filename, frame_bgr)

        # Create pygame surface for display (BGR -> RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.snapshot_surface = pygame.image.frombuffer(
            frame_rgb.tobytes(),
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            "RGB"
        )
        self.snapshot_file = snapshot_filename

        # Update leaderboard
        self.add_to_leaderboard(self.score, snapshot_filename)

    # -------------
    # Main Loop
    # -------------

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.frame_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_a:
                        self.state = "DEBUG"
                    elif self.state == "MODE_SELECT":
                        pass
                    elif self.state == "DEBUG":
                        if event.key == pygame.K_SPACE:
                            self.state = "MODE_SELECT"
                    elif self.state == "GAME_OVER" and event.key == pygame.K_SPACE:
                        self.reset_game()

            ret, frame_bgr = self.cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            hands_info = []
            if self.state in ("START", "RUNNING", "DEBUG"):
                if self.frame_count % 3 == 0:
                    hands_info = self.detector.detect(frame_rgb, WINDOW_WIDTH, WINDOW_HEIGHT)
                    self.last_hands_info = hands_info
                else:
                    hands_info = self.last_hands_info
            
            finger_info = None
            if self.state in ("MODE_SELECT", "MATH_READY", "MATH_RUNNING", "FRUIT_MODE_SELECT", "FRUIT_QUESTION", "DEBUG"):
                if self.frame_count % 3 == 0:
                    finger_info = self.detector.detect_finger_count(frame_rgb, WINDOW_WIDTH, WINDOW_HEIGHT)
                    if finger_info:
                        self.math_hand_up = finger_info["hand_up_flag"]
                        self.fruit_hand_up = finger_info["hand_up_flag"]
                        self.math_finger_count = finger_info["finger_count"]
                        self.fruit_finger_count = finger_info["finger_count"]
                        self.math_finger_stable = finger_info.get("stable", False)
                        self.math_current_finger = finger_info.get("current_count", finger_info["finger_count"])

            if self.state == "MODE_SELECT":
                self.update_mode_select_state()
            elif self.state == "START":
                self.update_start_state(hands_info)
            elif self.state == "READY":
                self.update_ready_state()
            elif self.state == "RUNNING":
                self.update_running_state(hands_info, frame_bgr)
            elif self.state == "MATH_START":
                self.update_math_start_state()
            elif self.state == "MATH_READY":
                self.update_math_ready_state()
            elif self.state == "MATH_RUNNING":
                self.update_math_running_state(frame_bgr)
            elif self.state == "FRUIT_MODE_SELECT":
                self.update_fruit_mode_select()
            elif self.state == "FRUIT_START":
                self.update_fruit_start_state()
            elif self.state == "FRUIT_DISPLAY":
                self.update_fruit_display_state()
            elif self.state == "FRUIT_QUESTION":
                self.update_fruit_question_state(frame_bgr)

            self.render(frame_rgb)

        self.cap.release()
        pygame.quit()
        sys.exit()

    # -------------
    # State Updates
    # -------------
    def update_mode_select_state(self):
        """模式选择状态 - 使用手指数量选择游戏模式"""
        if self.mode_select_time == 0:
            self.mode_select_time = time.time()
        
        # 检查手势复位（手放下）
        if self.mode_select_waiting_reset:
            if not self.math_hand_up:
                self.mode_select_waiting_reset = False
                print("Mode select: Gesture reset, ready for selection")
            return
        
        # 检查选择（必须检测到手才判断）
        if self.math_hand_up and self.math_finger_count in [1, 2, 3]:
            self.game_mode = self.math_finger_count
            print(f"Selected: Mode {self.math_finger_count}")
            self.mode_select_waiting_reset = True
            
            if self.math_finger_count == 1:
                self.state = "START"
            elif self.math_finger_count == 2:
                self.state = "MATH_START"
            elif self.math_finger_count == 3:
                self.state = "FRUIT_MODE_SELECT"
                self.fruit_mode_select_time = time.time()

    def update_start_state(self, hands_info):
        """
        START state: wait for the player to show SCISSORS gesture
        to begin the game.
        """
        for hand in hands_info:
            if hand["gesture"] == "scissors":
                self.state = "READY"
                self.ready_start_time = time.time() # 记录开始时间
                break

    def update_ready_state(self):
        # 检查是否过了 2 秒
        if time.time() - self.ready_start_time >= 2.0:
            self.state = "RUNNING"
            self.hp = HP_MAX
            self.score = 0
            self.icons = []
            self.last_spawn_time = time.time()

    def update_running_state(self, hands_info, frame_bgr):
        now = time.time()
        
        # 1. 刷怪逻辑 (保持不变)
        if (now - self.last_spawn_time >= SPAWN_INTERVAL and len(self.icons) < MAX_ICONS):
            self.spawn_icon()

        # 2. 超时扣血逻辑 (保持不变)
        remaining_icons = []
        for icon in self.icons:
            if now - icon.spawn_time > icon.lifetime:
                self.hp -= 1
                if self.hp <= 0 and self.state != "GAME_OVER":
                    self.trigger_game_over(frame_bgr)
                return
            else:
                remaining_icons.append(icon)
        self.icons = remaining_icons

        # 3. 核心碰撞检测逻辑 (只保留这一套)
        if not hands_info:
            return

        icons_to_remove = set()
        for hand in hands_info:
            # 只处理有效的象限
            if hand["quadrant"] == "None":
                continue
                
            for i, icon in enumerate(self.icons):
                if i in icons_to_remove:
                    continue

                # 判定条件：手所在的象限 == 图标所在的象限
                if hand["quadrant"] == icon.quadrant:
                    # 只有当手势不是 none 时才进行真正的胜负判定
                    if hand["gesture"] == "none":
                        continue
                    
                    print(f"DEBUG: Hit Quadrant! Icon[{icon.required_handedness}, {icon.gesture}] vs Hand[{hand['handedness']}, {hand['gesture']}]")
                    
                    # 1) 左右手校验
                    if icon.required_handedness != hand["handedness"]:
                        print(" -> 手错误!")
                        self.hp -= 1
                        icons_to_remove.add(i)
                    else:
                        # 2) 手势校验
                        needed_gesture = Icon.counter_gesture(icon.gesture)
                        if hand["gesture"] == needed_gesture:
                            print(" -> 得分!")
                            self.score += 1
                            icons_to_remove.add(i)
                        else:
                            print(" -> 手势错误!")
                            self.hp -= 1
                            icons_to_remove.add(i)

        # 4. 移除已处理图标
        self.icons = [icon for idx, icon in enumerate(self.icons) if idx not in icons_to_remove]
        
        # 5. 最后检查 HP
        if self.hp <= 0 and self.state != "GAME_OVER":
            self.trigger_game_over(frame_bgr)


        # 判定手势所在象限
        def get_quadrant_name(self, center_px):
            cx, cy = center_px
            # 将屏幕分为 4 个象限
            is_left = cx < WINDOW_WIDTH // 2
            is_top = cy < WINDOW_HEIGHT // 2
            
            if is_left and is_top: return "Top-Left"
            if not is_left and is_top: return "Top-Right"
            if is_left and not is_top: return "Bottom-Left"
            return "Bottom-Right"

    def generate_math_question(self):
        """生成简单的加减法题目，答案在0-5范围内"""
        answer = random.randint(0, 5)
        operation = random.choice(['+', '-'])
        
        if answer == 0:
            a = random.randint(0, 5)
            b = a
            self.math_question = f"{a} - {b} = ?"
        elif operation == '+':
            a = random.randint(0, answer)
            b = answer - a
            self.math_question = f"{a} + {b} = ?"
        else:
            a = answer + random.randint(0, 5 - answer)
            b = a - answer
            self.math_question = f"{a} - {b} = ?"
        
        self.math_answer = answer

    def update_math_start_state(self):
        """数学游戏开始状态"""
        self.math_question_num = 0
        self.math_correct_count = 0
        self.generate_math_question()
        self.math_question_time = time.time()
        self.state = "MATH_READY"
        self.ready_start_time = time.time()
        self.math_waiting_reset = True  # 强制要求先放下手

    def update_math_ready_state(self):
        """数学游戏准备状态"""
        if time.time() - self.ready_start_time >= 1.5:
            self.state = "MATH_RUNNING"
            self.math_question_time = time.time()

    def update_math_running_state(self, frame_bgr):
        """数学游戏运行状态"""
        now = time.time()
        
        # 检查手势复位（手放下）
        if self.math_waiting_reset:
            if not self.math_hand_up:
                self.math_waiting_reset = False
                print("Gesture reset, ready for next answer")
            return
        
        # 只有当手指数量稳定（连续4帧相同）时才判断答案
        if self.math_hand_up and self.math_finger_stable:
            if self.math_finger_count == self.math_answer:
                # 正确答案
                self.math_correct_count += 1
                print(f"Correct! Answer: {self.math_answer}")
                self.math_question_num += 1
                self.math_waiting_reset = True  # 需要手势复位
                
                if self.math_question_num >= self.math_total_questions:
                    self.score = self.math_correct_count
                    self.trigger_game_over(frame_bgr)
                else:
                    self.generate_math_question()
                    self.math_question_time = time.time()
            
            else:
                # 错误答案：继续下一题
                print(f"Wrong! Answer was: {self.math_answer}, you showed {self.math_finger_count}")
                self.math_question_num += 1
                self.math_waiting_reset = True  # 需要手势复位
                
                if self.math_question_num >= self.math_total_questions:
                    self.score = self.math_correct_count
                    self.trigger_game_over(frame_bgr)
                else:
                    self.generate_math_question()
                    self.math_question_time = time.time()
        
        elif now - self.math_question_time > self.math_time_limit:
            print(f"Timeout! Answer was: {self.math_answer}")
            self.math_question_num += 1
            
            if self.math_question_num >= self.math_total_questions:
                self.score = self.math_correct_count
                self.trigger_game_over(frame_bgr)
            else:
                self.generate_math_question()
                self.math_question_time = time.time()
                self.math_waiting_reset = True  # 需要手势复位

    def update_fruit_mode_select(self):
        """水果游戏模式选择状态"""
        # 检查手势复位（手放下）
        if self.fruit_mode_waiting_reset:
            if not self.fruit_hand_up:
                self.fruit_mode_waiting_reset = False
                print("Fruit mode select: Gesture reset")
            return
        
        if self.fruit_hand_up:
            if self.fruit_finger_count == 1:
                self.fruit_game_mode = 1  # 普通模式
                print("Selected: Normal Mode (5 questions, 2 fruit types)")
                self.fruit_mode_waiting_reset = True
                self.state = "FRUIT_START"
            elif self.fruit_finger_count == 2:
                self.fruit_game_mode = 2  # 挑战模式
                print("Selected: Challenge Mode (Difficulty increases, one mistake = game over)")
                self.fruit_mode_waiting_reset = True
                self.state = "FRUIT_START"

    def generate_fruit_question(self):
        """生成水果记忆游戏的问题"""
        available_fruits = ["Apple", "Banana", "Grapes", "Mango", "Pineapple", "Watermelon"]
        
        # 根据模式决定水果种类数
        if self.fruit_game_mode == 1:
            # 普通模式：固定2种水果
            num_types = 2
        else:
            # 挑战模式：根据难度递增
            num_types = min(self.fruit_num_types, len(available_fruits))
        
        self.fruit_types = random.sample(available_fruits, num_types)
        
        # 随机生成每种水果的数量（0-5），至少有一种水果数量>0
        self.fruit_counts = {}
        self.fruit_positions = []
        
        # 确保至少有一个水果
        guaranteed_fruit = random.choice(self.fruit_types)
        guaranteed_count = random.randint(1, 5)
        self.fruit_counts[guaranteed_fruit] = guaranteed_count
        
        for _ in range(guaranteed_count):
            x = random.randint(80, WINDOW_WIDTH - 80)
            y = random.randint(100, WINDOW_HEIGHT - 100)
            self.fruit_positions.append({"fruit": guaranteed_fruit, "x": x, "y": y})
        
        # 其他水果可以是0-5
        for fruit in self.fruit_types:
            if fruit == guaranteed_fruit:
                continue
            count = random.randint(0, 5)
            self.fruit_counts[fruit] = count
            
            for _ in range(count):
                x = random.randint(80, WINDOW_WIDTH - 80)
                y = random.randint(100, WINDOW_HEIGHT - 100)
                self.fruit_positions.append({"fruit": fruit, "x": x, "y": y})
        
        # 随机选择要问的水果（可以问数量为0的水果）
        self.fruit_target = random.choice(self.fruit_types)
        self.fruit_answer = self.fruit_counts[self.fruit_target]

    def update_fruit_start_state(self):
        """水果记忆游戏开始状态"""
        self.fruit_question_num = 0
        self.fruit_correct_count = 0
        self.fruit_consecutive_correct = 0
        self.fruit_num_types = 2
        self.fruit_waiting_reset = True  # 强制要求先放下手
        self.generate_fruit_question()
        self.fruit_question_time = time.time()
        self.state = "FRUIT_DISPLAY"

    def update_fruit_display_state(self):
        """水果显示状态"""
        if time.time() - self.fruit_question_time >= self.fruit_display_time:
            self.state = "FRUIT_QUESTION"
            self.fruit_question_time = time.time()
            self.fruit_waiting_reset = True  # 强制要求先放下手

    def update_fruit_question_state(self, frame_bgr):
        """水果问题回答状态"""
        now = time.time()
        
        # 检查手势复位（手放下）
        if self.fruit_waiting_reset:
            if not self.fruit_hand_up:
                self.fruit_waiting_reset = False
                print("Gesture reset, ready for next answer")
            return
        
        # 检查答案（必须检测到手才判断，答案可以是0）
        if self.fruit_hand_up and self.fruit_finger_count == self.fruit_answer:
            self.fruit_correct_count += 1
            if self.fruit_game_mode == 2:
                self.fruit_consecutive_correct += 1
            print(f"Correct! There are {self.fruit_answer} {self.fruit_target}s")
            self.fruit_question_num += 1
            self.fruit_waiting_reset = True  # 需要手势复位
            
            # 挑战模式：每3题正确增加一种水果
            if self.fruit_game_mode == 2:
                if self.fruit_consecutive_correct > 0 and self.fruit_consecutive_correct % 3 == 0:
                    if self.fruit_num_types < 6:
                        self.fruit_num_types += 1
                        print(f"Difficulty increased! Now {self.fruit_num_types} fruit types")
            
            # 检查是否达到上限
            max_q = self.fruit_total_questions if self.fruit_game_mode == 1 else self.fruit_max_questions
            if self.fruit_question_num >= max_q:
                self.score = self.fruit_correct_count
                self.trigger_game_over(frame_bgr)
            else:
                self.generate_fruit_question()
                self.fruit_question_time = time.time()
                self.state = "FRUIT_DISPLAY"
        
        elif self.fruit_hand_up and self.fruit_finger_count != self.fruit_answer:
            # 错误答案：立即判断
            print(f"Wrong! There were {self.fruit_answer} {self.fruit_target}s, not {self.fruit_finger_count}")
            
            if self.fruit_game_mode == 1:
                # 普通模式：继续下一题
                self.fruit_question_num += 1
                if self.fruit_question_num >= self.fruit_total_questions:
                    self.score = self.fruit_correct_count
                    self.trigger_game_over(frame_bgr)
                else:
                    self.generate_fruit_question()
                    self.fruit_question_time = time.time()
                    self.state = "FRUIT_DISPLAY"
            else:
                # 挑战模式：失败退出
                print("Game Over! One mistake and you're out.")
                self.score = self.fruit_correct_count
                self.trigger_game_over(frame_bgr)
        
        elif now - self.fruit_question_time > self.fruit_answer_time:
            print(f"Timeout! There were {self.fruit_answer} {self.fruit_target}s")
            
            if self.fruit_game_mode == 1:
                # 普通模式：继续下一题
                self.fruit_question_num += 1
                if self.fruit_question_num >= self.fruit_total_questions:
                    self.score = self.fruit_correct_count
                    self.trigger_game_over(frame_bgr)
                else:
                    self.generate_fruit_question()
                    self.fruit_question_time = time.time()
                    self.state = "FRUIT_DISPLAY"
            else:
                # 挑战模式：失败退出
                print("Game Over! One mistake and you're out.")
                self.score = self.fruit_correct_count
                self.trigger_game_over(frame_bgr)

    # -------------
    # Rendering
    # -------------

    def render(self, frame_rgb):
        frame_surface = pygame.image.frombuffer(
            frame_rgb.tobytes(),
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            "RGB"
        )

        self.screen.blit(frame_surface, (0, 0))

        if self.state == "MODE_SELECT":
            self.draw_mode_select_overlay()
        elif self.state == "DEBUG":
            self.draw_debug_overlay()
        elif self.state == "START":
            self.draw_hud()
            self.draw_start_overlay()
        elif self.state == "READY":
            self.draw_hud()
            self.draw_ready_overlay()
        elif self.state == "RUNNING":
            self.draw_icons()
            self.draw_hud()
        elif self.state == "MATH_START":
            self.draw_math_start_overlay()
        elif self.state == "MATH_READY":
            self.draw_math_ready_overlay()
        elif self.state == "MATH_RUNNING":
            self.draw_math_game_ui()
        elif self.state == "FRUIT_MODE_SELECT":
            self.draw_fruit_mode_select_overlay()
        elif self.state == "FRUIT_START":
            self.draw_fruit_start_overlay()
        elif self.state == "FRUIT_DISPLAY":
            self.draw_fruit_display_ui()
        elif self.state == "FRUIT_QUESTION":
            self.draw_fruit_question_ui()
        elif self.state == "GAME_OVER":
            self.draw_game_over()

        pygame.display.flip()

    def draw_ready_overlay(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        # 显示倒计时逻辑
        elapsed = time.time() - self.ready_start_time
        countdown = max(0, int(2.1 - elapsed)) # 显示 2, 1, 0
        
        text = self.big_font.render(f"Get Ready: {countdown}", True, (255, 255, 0))
        text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(text, text_rect)

    def draw_mode_select_overlay(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        title = self.big_font.render("Select Game Mode", True, (255, 255, 255))
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100))
        self.screen.blit(title, title_rect)
        
        mode1 = self.font.render("Show 1 finger: Rock-Paper-Scissors Battle", True, (100, 255, 100))
        mode1_rect = mode1.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 40))
        self.screen.blit(mode1, mode1_rect)
        
        mode2 = self.font.render("Show 2 fingers: Math Finger Counting", True, (255, 200, 100))
        mode2_rect = mode2.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(mode2, mode2_rect)
        
        mode3 = self.font.render("Show 3 fingers: Fruit Memory Game", True, (100, 200, 255))
        mode3_rect = mode3.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 40))
        self.screen.blit(mode3, mode3_rect)
        
        if self.math_hand_up:
            finger_text = self.font.render(f"Detected: {self.math_finger_count} finger(s)", True, (255, 255, 0))
        else:
            finger_text = self.font.render("Detected: Show your hand", True, (150, 150, 150))
        finger_rect = finger_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 80))
        self.screen.blit(finger_text, finger_rect)
        
        if self.mode_select_waiting_reset:
            reset_hint = self.font.render("Put your hand down to confirm", True, (255, 255, 0))
            reset_rect = reset_hint.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 110))
            self.screen.blit(reset_hint, reset_rect)
        
        hint = self.font.render("ESC to quit", True, (150, 150, 150))
        hint_rect = hint.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 140))
        self.screen.blit(hint, hint_rect)

    def draw_debug_overlay(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 80))
        self.screen.blit(overlay, (0, 0))
        
        title = self.big_font.render("DEBUG MODE", True, (255, 255, 0))
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(title, title_rect)
        
        hint = self.font.render("Press SPACE to exit", True, (150, 150, 150))
        hint_rect = hint.get_rect(center=(WINDOW_WIDTH // 2, 60))
        self.screen.blit(hint, hint_rect)
        
        y_offset = 100
        
        left_gesture = "none"
        right_gesture = "none"
        
        for hand in self.last_hands_info:
            if hand["handedness"] == "Left":
                left_gesture = hand["gesture"]
            elif hand["handedness"] == "Right":
                right_gesture = hand["gesture"]
        
        left_text = self.font.render(f"Left Hand: {left_gesture}", True, (255, 100, 100))
        left_rect = left_text.get_rect(center=(WINDOW_WIDTH // 2, y_offset))
        self.screen.blit(left_text, left_rect)
        
        right_text = self.font.render(f"Right Hand: {right_gesture}", True, (100, 255, 100))
        right_rect = right_text.get_rect(center=(WINDOW_WIDTH // 2, y_offset + 30))
        self.screen.blit(right_text, right_rect)
        
        finger_y = y_offset + 70
        if self.math_hand_up:
            finger_text = self.font.render(f"Finger Count: {self.math_finger_count}", True, (255, 255, 255))
        else:
            finger_text = self.font.render("Finger Count: No hand detected", True, (255, 100, 100))
        finger_rect = finger_text.get_rect(center=(WINDOW_WIDTH // 2, finger_y))
        self.screen.blit(finger_text, finger_rect)
        
        for hand in self.last_hands_info:
            if hand["center"]:
                cx, cy = hand["center"]
                color = (255, 100, 100) if hand["handedness"] == "Left" else (100, 255, 100)
                pygame.draw.circle(self.screen, color, (cx, cy), 20, 3)
                label = self.font.render(hand["handedness"], True, color)
                label_rect = label.get_rect(center=(cx, cy - 30))
                self.screen.blit(label, label_rect)

    def draw_math_start_overlay(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        title = self.big_font.render("Math Finger Game", True, (255, 200, 100))
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 60))
        self.screen.blit(title, title_rect)
        
        instr = self.font.render("Show finger count to answer", True, (255, 255, 255))
        instr_rect = instr.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 20))
        self.screen.blit(instr, instr_rect)
        
        instr2 = self.font.render("Answer range: 1-5", True, (255, 255, 255))
        instr2_rect = instr2.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 10))
        self.screen.blit(instr2, instr2_rect)
        
        instr3 = self.font.render("10 questions, 10 seconds each", True, (255, 255, 255))
        instr3_rect = instr3.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 40))
        self.screen.blit(instr3, instr3_rect)

    def draw_math_ready_overlay(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        elapsed = time.time() - self.ready_start_time
        countdown = max(0, int(1.6 - elapsed))
        
        text = self.big_font.render(f"Starting in: {countdown}", True, (255, 255, 0))
        text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(text, text_rect)

    def draw_math_game_ui(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.screen.blit(overlay, (0, 0))
        
        question = self.big_font.render(self.math_question, True, (255, 255, 255))
        question_rect = question.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 80))
        self.screen.blit(question, question_rect)
        
        progress = self.font.render(f"Question: {self.math_question_num + 1}/{self.math_total_questions}", True, (255, 255, 255))
        progress_rect = progress.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(progress, progress_rect)
        
        correct = self.font.render(f"Correct: {self.math_correct_count}", True, (100, 255, 100))
        correct_rect = correct.get_rect(center=(WINDOW_WIDTH // 2, 60))
        self.screen.blit(correct, correct_rect)
        
        elapsed = time.time() - self.math_question_time
        remaining = max(0, int(self.math_time_limit - elapsed))
        timer = self.font.render(f"Time: {remaining}s", True, (255, 200, 100) if remaining > 3 else (255, 50, 50))
        timer_rect = timer.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
        self.screen.blit(timer, timer_rect)
        
        # 显示当前检测到的手指数和稳定状态
        if self.math_hand_up:
            if self.math_finger_stable:
                # 稳定：显示绿色确认标记
                finger_text = self.font.render(f"Fingers: {self.math_finger_count} ✓", True, (100, 255, 100))
                stable_text = self.font.render("Confirmed!", True, (100, 255, 100))
            else:
                # 不稳定：显示当前检测到的数量（黄色）
                finger_text = self.font.render(f"Detecting: {self.math_current_finger}", True, (255, 255, 0))
                stable_text = self.font.render("Hold steady...", True, (255, 255, 0))
        else:
            finger_text = self.font.render("Fingers: Show your hand", True, (150, 150, 150))
            stable_text = self.font.render("", True, (150, 150, 150))
        
        finger_rect = finger_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(finger_text, finger_rect)
        
        stable_rect = stable_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 30))
        self.screen.blit(stable_text, stable_rect)
        
        # 显示复位提示
        if self.math_waiting_reset:
            reset_hint = self.font.render("Put your hand down to continue", True, (255, 255, 0))
            reset_rect = reset_hint.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 60))
            self.screen.blit(reset_hint, reset_rect)

    def draw_fruit_mode_select_overlay(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        title = self.big_font.render("Fruit Memory Game", True, (255, 200, 100))
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100))
        self.screen.blit(title, title_rect)
        
        select = self.font.render("Select Game Mode", True, (255, 255, 255))
        select_rect = select.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
        self.screen.blit(select, select_rect)
        
        mode1 = self.font.render("Show 1 finger: Normal Mode", True, (100, 255, 100))
        mode1_rect = mode1.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(mode1, mode1_rect)
        
        mode1_desc = self.font.render("5 questions, 2 fruit types", True, (200, 200, 200))
        mode1_desc_rect = mode1_desc.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 25))
        self.screen.blit(mode1_desc, mode1_desc_rect)
        
        mode2 = self.font.render("Show 2 fingers: Challenge Mode", True, (255, 200, 100))
        mode2_rect = mode2.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 60))
        self.screen.blit(mode2, mode2_rect)
        
        mode2_desc = self.font.render("Difficulty increases, one mistake = game over", True, (200, 200, 200))
        mode2_desc_rect = mode2_desc.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 85))
        self.screen.blit(mode2_desc, mode2_desc_rect)
        
        if self.fruit_hand_up:
            finger = self.font.render(f"Detected: {self.fruit_finger_count} finger(s)", True, (255, 255, 0))
        else:
            finger = self.font.render("Detected: No hand", True, (255, 100, 100))
        finger_rect = finger.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 130))
        self.screen.blit(finger, finger_rect)

    def draw_fruit_start_overlay(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        if self.fruit_game_mode == 1:
            title = self.big_font.render("Normal Mode", True, (100, 255, 100))
            title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 60))
            self.screen.blit(title, title_rect)
            
            instr1 = self.font.render("5 questions, 2 fruit types", True, (255, 255, 255))
            instr1_rect = instr1.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 20))
            self.screen.blit(instr1, instr1_rect)
            
            instr2 = self.font.render("Answer can be 0-5", True, (255, 255, 255))
            instr2_rect = instr2.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 10))
            self.screen.blit(instr2, instr2_rect)
            
            instr3 = self.font.render("Wrong answers continue to next question", True, (255, 255, 255))
            instr3_rect = instr3.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 40))
            self.screen.blit(instr3, instr3_rect)
        else:
            title = self.big_font.render("Challenge Mode", True, (255, 200, 100))
            title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 80))
            self.screen.blit(title, title_rect)
            
            instr1 = self.font.render("Memorize the number of each fruit", True, (255, 255, 255))
            instr1_rect = instr1.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 40))
            self.screen.blit(instr1, instr1_rect)
            
            instr2 = self.font.render("Answer can be 0-5", True, (255, 255, 255))
            instr2_rect = instr2.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 10))
            self.screen.blit(instr2, instr2_rect)
            
            instr3 = self.font.render("One mistake = Game Over!", True, (255, 100, 100))
            instr3_rect = instr3.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
            self.screen.blit(instr3, instr3_rect)
            
            instr4 = self.font.render("Every 3 correct = More fruit types", True, (255, 255, 0))
            instr4_rect = instr4.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
            self.screen.blit(instr4, instr4_rect)
            
            instr5 = self.font.render("Max 9 questions", True, (255, 255, 255))
            instr5_rect = instr5.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 80))
            self.screen.blit(instr5, instr5_rect)
        
        start = self.font.render("Starting...", True, (255, 255, 0))
        start_rect = start.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 120))
        self.screen.blit(start, start_rect)

    def draw_fruit_display_ui(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 50))
        self.screen.blit(overlay, (0, 0))
        
        max_q = self.fruit_total_questions if self.fruit_game_mode == 1 else self.fruit_max_questions
        progress = self.font.render(f"Question: {self.fruit_question_num + 1}/{max_q}", True, (255, 255, 255))
        progress_rect = progress.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(progress, progress_rect)
        
        correct = self.font.render(f"Correct: {self.fruit_correct_count}", True, (100, 255, 100))
        correct_rect = correct.get_rect(center=(WINDOW_WIDTH // 2, 60))
        self.screen.blit(correct, correct_rect)
        
        # 挑战模式显示难度
        if self.fruit_game_mode == 2:
            difficulty = self.font.render(f"Fruit Types: {self.fruit_num_types}", True, (255, 200, 100))
            difficulty_rect = difficulty.get_rect(center=(WINDOW_WIDTH // 2, 90))
            self.screen.blit(difficulty, difficulty_rect)
        
        elapsed = time.time() - self.fruit_question_time
        remaining = max(0, int(self.fruit_display_time - elapsed))
        timer = self.font.render(f"Memorize: {remaining}s", True, (255, 255, 0))
        timer_rect = timer.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
        self.screen.blit(timer, timer_rect)
        
        fruit_size = 60
        for item in self.fruit_positions:
            fruit_img = self.fruit_images[item["fruit"]]
            scaled_img = pygame.transform.scale(fruit_img, (fruit_size, fruit_size))
            img_rect = scaled_img.get_rect(center=(item["x"], item["y"]))
            self.screen.blit(scaled_img, img_rect)

    def draw_fruit_question_ui(self):
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.screen.blit(overlay, (0, 0))
        
        max_q = self.fruit_total_questions if self.fruit_game_mode == 1 else self.fruit_max_questions
        progress = self.font.render(f"Question: {self.fruit_question_num + 1}/{max_q}", True, (255, 255, 255))
        progress_rect = progress.get_rect(center=(WINDOW_WIDTH // 2, 30))
        self.screen.blit(progress, progress_rect)
        
        correct = self.font.render(f"Correct: {self.fruit_correct_count}", True, (100, 255, 100))
        correct_rect = correct.get_rect(center=(WINDOW_WIDTH // 2, 60))
        self.screen.blit(correct, correct_rect)
        
        # 挑战模式显示难度
        if self.fruit_game_mode == 2:
            difficulty = self.font.render(f"Fruit Types: {self.fruit_num_types}", True, (255, 200, 100))
            difficulty_rect = difficulty.get_rect(center=(WINDOW_WIDTH // 2, 90))
            self.screen.blit(difficulty, difficulty_rect)
        
        question_text = f"How many {self.fruit_target}s were there?"
        question = self.big_font.render(question_text, True, (255, 255, 255))
        question_rect = question.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 60))
        self.screen.blit(question, question_rect)
        
        elapsed = time.time() - self.fruit_question_time
        remaining = max(0, int(self.fruit_answer_time - elapsed))
        timer = self.font.render(f"Time: {remaining}s", True, (255, 200, 100) if remaining > 3 else (255, 50, 50))
        timer_rect = timer.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
        self.screen.blit(timer, timer_rect)
        
        if self.fruit_hand_up:
            finger_text = self.font.render(f"Fingers: {self.fruit_finger_count}", True, (255, 255, 255))
        else:
            finger_text = self.font.render("Fingers: Show your hand", True, (150, 150, 150))
        finger_rect = finger_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
        self.screen.blit(finger_text, finger_rect)
        
        # 显示复位提示
        if self.fruit_waiting_reset:
            reset_hint = self.font.render("Put your hand down to continue", True, (255, 255, 0))
            reset_rect = reset_hint.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 60))
            self.screen.blit(reset_hint, reset_rect)

    def draw_gesture_icon(self, surface, gesture, center_x, center_y, size, bg_color):
        cx, cy = int(center_x), int(center_y)
        radius = size // 2 - 5
        
        pygame.draw.circle(surface, bg_color, (cx, cy), radius)
        pygame.draw.circle(surface, (255, 255, 255), (cx, cy), radius, 3)
        
        if gesture in self.gesture_images:
            img = pygame.transform.scale(self.gesture_images[gesture], (size - 10, size - 10))
            img_rect = img.get_rect(center=(cx, cy))
            surface.blit(img, img_rect)

    def draw_icons(self):
        """
        Draw icons (enemy gestures) on top of the camera feed.
        Each icon shows a hand gesture symbol.
        """
        for icon in self.icons:
            self.draw_gesture_icon(
                self.screen, 
                icon.gesture, 
                icon.rect.centerx, 
                icon.rect.centery, 
                ICON_SIZE,
                icon.color
            )

    def draw_hud(self):
        """
        Draw HP (top-left) and Score (top-right).
        """
        # HP
        hp_text = self.font.render(f"HP: {self.hp}", True, (255, 255, 255))
        self.screen.blit(hp_text, (10, 10))

        # Score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(topright=(WINDOW_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

    def draw_start_overlay(self):
        """
        Instructions overlay in START state.
        """
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))  # semi-transparent dark overlay
        self.screen.blit(overlay, (0, 0))

        title = self.big_font.render("Hand Gesture Battle", True, (255, 255, 255))
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 40))
        self.screen.blit(title, title_rect)

        instr1 = self.font.render("Show SCISSORS gesture to start.", True, (255, 255, 255))
        instr1_rect = instr1.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 10))
        self.screen.blit(instr1, instr1_rect)

        instr2 = self.font.render("Hit icons with the correct counter gesture.", True, (255, 255, 255))
        instr2_rect = instr2.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 40))
        self.screen.blit(instr2, instr2_rect)

    def draw_game_over(self):
        """ GAME_OVER screen: - Dark overlay - Show snapshot thumbnail - Show final score - Show leaderboard - Press SPACE to restart """
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # 1. 标题 (Y=40)
        title = self.big_font.render("GAME OVER", True, (255, 50, 50))
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 40))
        self.screen.blit(title, title_rect)

        # 2. 分数 (Y=80)
        final_score = self.font.render(f"Your Score: {self.score}", True, (255, 255, 255))
        final_score_rect = final_score.get_rect(center=(WINDOW_WIDTH // 2, 80))
        self.screen.blit(final_score, final_score_rect)

        # 3. 截图缩略图 (中心 Y=200，图片高度约 150px，底部约 275)
        if self.snapshot_surface is not None:
            thumb_width = 200
            thumb_height = int(thumb_width * WINDOW_HEIGHT / WINDOW_WIDTH)
            thumb = pygame.transform.smoothscale(self.snapshot_surface, (thumb_width, thumb_height))
            thumb_rect = thumb.get_rect(center=(WINDOW_WIDTH // 2, 200))
            self.screen.blit(thumb, thumb_rect)

        # 4. 排行榜 (Y=290 开始)
        lb_title = self.font.render("Leaderboard (Top 5)", True, (255, 255, 0))
        lb_title_rect = lb_title.get_rect(center=(WINDOW_WIDTH // 2, 290))
        self.screen.blit(lb_title, lb_title_rect)

        y_start = 320 # 排行榜列表起始位置
        for i, entry in enumerate(self.leaderboard[:5]):
            line = f"{i + 1}. {entry.get('name', 'Player')} - {entry.get('score', 0)}"
            text = self.font.render(line, True, (255, 255, 255))
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, y_start + i * 25))
            self.screen.blit(text, text_rect)

        # 5. 底部提示
        instr = self.font.render("Press SPACE to restart, ESC to quit.", True, (255, 255, 255))
        instr_rect = instr.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
        self.screen.blit(instr, instr_rect)


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    game = HandGestureBattleGame()
    game.run()
