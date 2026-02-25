# HandGestureBattleGame
A webcam-based Natural User Interface (NUI) game. Players throw Rock-Paper-Scissors gestures to hit icons in four quadrants—win to score, miss to lose HP. When HP hits zero, a photo is taken for the leaderboard. 

### Technical Design Document (TDD)
#### System Architecture
- Input: Capture real-time video frames (640x480) via OpenCV.
Processing:
Use MediaPipe Hand Landmarker to get 21 landmarks and Handedness (Left/Right labels).
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

Initialize OpenCV video stream and integrate MediaPipe.
Develop a GestureDetector to identify Rock, Paper, and Scissors.

#### Phase 2: Game Engine (Logic & UI)

Create the Pygame main loop and background rendering.
Implement the Random Spawn System for icons.
Build the HP (3 Lives) and Scoring system.

#### Phase 3: Features & Polish

Add Handedness Logic: Red icons for the left hand, Green for the right hand.
Implement the Auto-Snapshot feature when the game ends.
Design the Leaderboard UI to display player photos and scores.

#### Phase 4: Testing

Conduct a Design Review and user testing to calibrate gesture sensitivity.
Fix coordinate mirroring and offset issues.