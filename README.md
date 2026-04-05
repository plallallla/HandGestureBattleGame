# HandGestureBattleGame / 手势对战游戏

[English](#english) | [中文](#中文)

---

<a name="english"></a>

A webcam-based Natural User Interface (NUI) game collection featuring three exciting game modes with hand gesture recognition.

### 🎮 Game Modes

#### Mode 1: Rock-Paper-Scissors Battle
**How to Start:**
- Show **SCISSORS** gesture → Normal Battle Mode
- Show **ROCK** gesture → Bully Mode

**Normal Battle Mode:**
- Icons appear in four quadrants of the screen
- Red icons require **Left Hand**, Green icons require **Right Hand**
- Show the correct counter-gesture to destroy icons
  - Rock beats Scissors
  - Paper beats Rock
  - Scissors beats Paper
- 3 HP (lives): Lose 1 HP when timeout or wrong gesture
- Survive as long as possible to maximize score

**Bully Mode:**
- One icon appears per round (10 rounds total)
- System gives requirement: **WIN**, **DRAW**, or **LOSE**
- Show the correct gesture according to requirement
- 5 seconds per round
- Score based on correct answers

#### Mode 2: Math Finger Counting
**How to Play:**
- Answer 10 simple math questions (addition/subtraction)
- Use your fingers to show the answer (0-5)
- 10 seconds per question
- Gesture must be stable for 4 consecutive frames to confirm
- UI Indicators:
  - Yellow "Detecting": Gesture not yet stable
  - Green "Confirmed ✓": Gesture confirmed, answer submitted

#### Mode 3: Fruit Memory Game
**How to Start:**
- Show **1 finger** → Normal Mode
- Show **2 fingers** → Challenge Mode

**Normal Mode:**
- 5 questions, 2 fruit types
- Memorize the count of each fruit type displayed (5 seconds)
- Answer by showing finger count (0-5)
- Wrong answers continue to next question
- 10 seconds to answer

**Challenge Mode:**
- Difficulty increases every 3 correct answers (up to 6 fruit types)
- One mistake = Game Over
- Maximum 9 questions
- Higher difficulty = More fruit types to memorize

### 🎯 Controls

#### Mode Selection
- **1 finger**: Rock-Paper-Scissors Battle
- **2 fingers**: Math Finger Counting
- **3 fingers**: Fruit Memory Game

#### Game Controls
- **Put hand down → Raise hand**: Required gesture sequence to confirm answers
- **ESC**: Quit game
- **SPACE**: Restart game (after Game Over)

### 🏆 Leaderboard System
- Separate leaderboard for each game mode
- Top 5 scores displayed on Game Over screen
- Snapshot saved automatically when game ends
- Scores persist in `leaderboard.json`

### 📋 Gesture Recognition
- **Rock**: Close all fingers
- **Paper**: Open all fingers
- **Scissors**: Index and middle finger extended
- **Finger Count**: Count extended fingers (0-5)
- Stability: 4 consecutive frames with same gesture required for confirmation

---

<a name="中文"></a>

基于摄像头的自然用户界面（NUI）游戏合集，包含三种刺激的游戏模式和手势识别功能。

### 🎮 游戏模式

#### 模式1：石头剪刀布对战
**启动方式：**
- 比出**剪刀**手势 → 普通对战模式
- 比出**石头**手势 → 斗牛模式

**普通对战模式：**
- 图标出现在屏幕四个象限
- 红色图标需要用**左手**，绿色图标需要用**右手**
- 比出正确的克制手势消灭图标
  - 石头胜剪刀
  - 布胜石头
  - 剪刀胜布
- 3条生命：超时或手势错误扣1条命
- 尽可能长时间生存，获得最高分数

**斗牛模式：**
- 每轮出现1个图标（共10轮）
- 系统给出要求：**赢**、**平局** 或 **输**
- 根据要求展示正确手势
- 每轮限时5秒
- 按正确数量计分

#### 模式2：数学手指计数
**玩法：**
- 回答10道简单数学题（加减法）
- 用手指比出答案（0-5）
- 每题限时10秒
- 手势需连续4帧稳定才能确认
- 界面提示：
  - 黄色"Detecting"：手势尚未稳定
  - 绿色"Confirmed ✓"：手势已确认，答案提交

#### 模式3：水果记忆游戏
**启动方式：**
- 比出**1根手指** → 普通模式
- 比出**2根手指** → 挑战模式

**普通模式：**
- 5道题，2种水果
- 记忆显示的每种水果数量（5秒）
- 用手指比出答案（0-5）
- 答错继续下一题
- 回答时间10秒

**挑战模式：**
- 每3题正确增加难度（最多6种水果）
- 一次错误即游戏结束
- 最多9题
- 难度越高 = 需要记忆的水果种类越多

### 🎯 控制方式

#### 模式选择
- **1根手指**：石头剪刀布对战
- **2根手指**：数学手指计数
- **3根手指**：水果记忆游戏

#### 游戏操作
- **放下手 → 举起手**：确认答案所需的手势序列
- **ESC**：退出游戏
- **空格键**：重新开始游戏（游戏结束后）

### 🏆 排行榜系统
- 每个游戏模式独立排行榜
- 游戏结束时显示前5名分数
- 自动保存游戏结束时的截图
- 分数保存在 `leaderboard.json`

### 📋 手势识别
- **石头**：握拳
- **布**：张开所有手指
- **剪刀**：伸出食指和中指
- **手指计数**：计数伸出的手指数（0-5）
- 稳定性：需要连续4帧相同手势才能确认

---

### Technical Design Document (TDD)
#### System Architecture
- Input: Capture real-time video frames (640x480) via OpenCV.
- Processing:
  - Use MediaPipe Hand Landmarker to get 21 landmarks and Handedness (Left/Right labels).
  - Coordinate Mapping: Multiply MediaPipe's normalized coordinates (0.0 - 1.0) by the Pygame window size to get pixel positions.
  - Logic:
    - Gesture Recognition: Compare finger tip positions relative to knuckles to identify Rock, Paper, or Scissors.
    - Collision Detection: Check if the hand center overlaps with the target icon's hit-box.
- Output: Pygame renders the camera feed, game UI, animations, and life points (HP).

#### Core Game Logic
- Score: Triggered if Hand Position == Icon Position AND Hand Gesture == Counter-Gesture.
- Damage: Lose 1 HP if the icon disappears before being hit or if the wrong gesture is used.
- Game Over: When HP reaches 0, the system triggers a camera snapshot for the leaderboard.

### Development Task Breakdown (TODO)
#### Phase 1: Foundation (Camera & Tracking)
- Initialize OpenCV video stream and integrate MediaPipe.
- Develop a GestureDetector to identify Rock, Paper, and Scissors.

#### Phase 2: Game Engine (Logic & UI)
- Create the Pygame main loop and background rendering.
- Implement the Random Spawn System for icons.
- Build the HP (3 Lives) and Scoring system.

#### Phase 3: Features & Polish
- Add Handedness Logic: Red icons for the left hand, Green for the right hand.
- Implement the Auto-Snapshot feature when the game ends.
- Design the Leaderboard UI to display player photos and scores.

#### Phase 4: Testing
- Conduct a Design Review and user testing to calibrate gesture sensitivity.
- Fix coordinate mirroring and offset issues.