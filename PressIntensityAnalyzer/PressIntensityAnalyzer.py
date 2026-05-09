import numpy as np
import cv2
from collections import deque


class PressIntensityAnalyzer:
    """
    Measures how aggressively a team is pressing the opponent.

    Intensity = weighted combination of:
      1. Number of pressing-team players within PRESS_RADIUS of the ball
      2. How close those players are (closer = higher intensity)
      3. Their movement speed toward the ball

    Scale: 0.0 (no press) → 1.0 (maximum press)
    """

    PRESS_RADIUS_PX  = 250   # pixels — zone counted as pressing
    WINDOW_FRAMES    = 45    # rolling average window
    MIN_PRESSERS     = 1     # lowered to 1 so single presser still registers

    def __init__(self):
        self._history        = {1: deque(maxlen=self.WINDOW_FRAMES),
                                2: deque(maxlen=self.WINDOW_FRAMES)}
        self.intensity       = {1: 0.0, 2: 0.0}
        self._prev_positions = {}

    # ------------------------------------------------------------------

    def _get_ball_info(self, tracks, frame_num):
        """Returns (ball_pos, ball_holder_team)."""
        players_list = tracks.get("players", [])
        ball_list    = tracks.get("ball", [])

        # 1 — has_ball flag
        if frame_num < len(players_list):
            for _, info in players_list[frame_num].items():
                if info.get("has_ball"):
                    bbox = info.get("bbox", [])
                    if len(bbox) == 4:
                        return (
                            (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)),
                            info.get("team")
                        )

        # 2 — fallback: ball track + closest player
        if frame_num < len(ball_list):
            bbox = ball_list[frame_num].get(1, {}).get("bbox", [])
            if len(bbox) == 4:
                bx = int((bbox[0]+bbox[2])/2)
                by = int((bbox[1]+bbox[3])/2)

                min_dist     = float('inf')
                closest_team = None

                if frame_num < len(players_list):
                    for _, info in players_list[frame_num].items():
                        pbbox = info.get("bbox")
                        if pbbox is None:
                            continue
                        px = int((pbbox[0]+pbbox[2])/2)
                        py = int((pbbox[1]+pbbox[3])/2)
                        d  = np.hypot(px - bx, py - by)
                        if d < min_dist:
                            min_dist     = d
                            closest_team = info.get("team")

                return (bx, by), closest_team

        return None, None

    # ------------------------------------------------------------------

    def update(self, tracks, frame_num):
        players_list = tracks.get("players", [])
        if frame_num >= len(players_list):
            return

        ball_pos, ball_holder_team = self._get_ball_info(tracks, frame_num)
        if ball_pos is None:
            return

        players_in_frame = players_list[frame_num]

        for pressing_team in [1, 2]:
            if pressing_team == ball_holder_team:
                self._history[pressing_team].append(0.0)
                # still update positions
                for track_id, info in players_in_frame.items():
                    bbox = info.get("bbox")
                    if bbox:
                        self._prev_positions[track_id] = (
                            int((bbox[0]+bbox[2])/2),
                            int((bbox[1]+bbox[3])/2)
                        )
                continue

            scores = []

            for track_id, info in players_in_frame.items():
                if info.get("team") != pressing_team:
                    continue
                bbox = info.get("bbox")
                if bbox is None:
                    continue

                px   = int((bbox[0]+bbox[2])/2)
                py   = int((bbox[1]+bbox[3])/2)
                dist = np.hypot(px - ball_pos[0], py - ball_pos[1])

                if dist > self.PRESS_RADIUS_PX:
                    continue

                # Proximity score: 1.0 when on top of ball, 0.0 at edge of radius
                proximity_score = 1.0 - (dist / self.PRESS_RADIUS_PX)

                # Speed score: how fast moving (toward ball or generally)
                prev  = self._prev_positions.get(track_id)
                if prev is not None:
                    speed = np.hypot(px - prev[0], py - prev[1])
                    # Normalise: 10 px/frame = fast sprint
                    speed_score = min(1.0, speed / 10.0)
                else:
                    speed_score = 0.0

                # Combined score for this presser (proximity weighted more)
                player_score = 0.7 * proximity_score + 0.3 * speed_score
                scores.append(player_score)

            # Update previous positions
            for track_id, info in players_in_frame.items():
                bbox = info.get("bbox")
                if bbox:
                    self._prev_positions[track_id] = (
                        int((bbox[0]+bbox[2])/2),
                        int((bbox[1]+bbox[3])/2)
                    )

            if len(scores) < self.MIN_PRESSERS:
                self._history[pressing_team].append(0.0)
            else:
                # Average score × presser count factor
                avg_score      = float(np.mean(scores))
                presser_factor = min(1.0, len(scores) / 4.0)
                frame_intensity = avg_score * (0.6 + 0.4 * presser_factor)
                self._history[pressing_team].append(min(1.0, frame_intensity))

        # Smooth intensity
        for team_id in [1, 2]:
            h = self._history[team_id]
            self.intensity[team_id] = float(np.mean(h)) if h else 0.0

    # ------------------------------------------------------------------

    def draw(self, frame, team_colors=None):
        if team_colors is None:
            team_colors = {1: (0, 220, 100), 2: (0, 100, 220)}

        h, w    = frame.shape[:2]
        bar_w   = 200
        bar_h   = 18
        padding = 10

        for i, team_id in enumerate([1, 2]):
            x0        = padding
            y0        = h - padding - (i + 1) * (bar_h + 8)
            intensity = self.intensity[team_id]
            color     = team_colors.get(team_id, (200, 200, 200))

            # Background
            cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h),
                          (50, 50, 50), -1)
            # Fill
            fill_w = int(bar_w * intensity)
            if fill_w > 0:
                cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h),
                              color, -1)
            # Border
            cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h),
                          (200, 200, 200), 1)
            # Label
            cv2.putText(frame, f"T{team_id} Press: {intensity*100:.0f}%",
                        (x0 + bar_w + 8, y0 + bar_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        return frame

    def get_summary(self):
        return {
            f"team_{t}": {
                "avg_intensity": round(self.intensity[t], 3),
                "intensity_pct": round(self.intensity[t] * 100, 1),
            }
            for t in [1, 2]
        }