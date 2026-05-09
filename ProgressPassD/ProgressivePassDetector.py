import numpy as np
import cv2
from collections import defaultdict


class ProgressivePassDetector:
    """
    Detects progressive passes — passes that advance the ball
    significantly toward the opponent's goal.

    Definition used (adapted from StatsBomb / Opta):
        A pass is progressive if it moves the ball at least 25% of
        the remaining distance to the opponent's goal line.

    Also tracks:
        - Total progressive passes per team
        - Progressive pass success rate
        - Most progressive passers per team
    """

    # Minimum x-advance (pixels) to count as progressive
    MIN_ADVANCE_PX    = 80
    # Fraction of remaining distance to goal that must be covered
    PROGRESS_RATIO    = 0.25
    # Max frames between pass start and receive to count as one pass
    PASS_WINDOW       = 45

    def __init__(self, frame_width, attacking_direction=None):
        """
        Args:
            frame_width:         int — used to determine goal line x
            attacking_direction: dict {team_id: 'left'|'right'} 
                                 If None, team 1 attacks right, team 2 attacks left
        """
        self.frame_width = frame_width

        if attacking_direction is None:
            self.attack_dir = {1: "right", 2: "left"}
        else:
            self.attack_dir = attacking_direction

        # Records
        self.progressive_passes = defaultdict(list)
        # {team_id: [{"from": (x,y), "to": (x,y), "passer": id, "receiver": id}, ...]}

        self._pass_counts     = defaultdict(int)   # team_id -> total prog passes
        self._total_passes    = defaultdict(int)   # team_id -> all passes

        # Passer stats
        self._player_prog_passes = defaultdict(int)  # track_id -> prog pass count

        # Internal state
        self._last_holder       = {1: None, 2: None}
        self._last_holder_pos   = {1: None, 2: None}
        self._last_holder_frame = {1: -999, 2: -999}

        # For drawing — store recent progressive passes with fade
        self._draw_buffer = []   # [(from_pos, to_pos, color, frame_added)]
        self.FADE_FRAMES  = 90

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tracks, frame_num):
        """Call once per frame."""
        players_list = tracks.get("players", [])
        if frame_num >= len(players_list):
            return

        players_in_frame = players_list[frame_num]

        for track_id, info in players_in_frame.items():
            if not info.get("has_ball"):
                continue

            team_id = info.get("team")
            if team_id is None:
                continue

            bbox = info.get("bbox")
            if bbox is None:
                continue

            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            pos = (cx, cy)

            prev_id    = self._last_holder[team_id]
            prev_pos   = self._last_holder_pos[team_id]
            prev_frame = self._last_holder_frame[team_id]

            if (
                prev_id is not None
                and prev_id != track_id
                and (frame_num - prev_frame) <= self.PASS_WINDOW
                and prev_pos is not None
            ):
                # A pass occurred: prev_id → track_id
                self._total_passes[team_id] += 1
                if self._is_progressive(prev_pos, pos, team_id):
                    self._pass_counts[team_id] += 1
                    self._player_prog_passes[prev_id] += 1
                    self.progressive_passes[team_id].append({
                        "from": prev_pos,
                        "to": pos,
                        "passer": prev_id,
                        "receiver": track_id,
                        "frame": frame_num,
                    })
                    # Add to draw buffer
                    color = (0, 255, 180) if team_id == 1 else (255, 180, 0)
                    self._draw_buffer.append((prev_pos, pos, color, frame_num))

            self._last_holder[team_id]       = track_id
            # self._last_holder_pos[team_id]   = pos
            self._last_holder_frame[team_id] = frame_num
            break

    def draw(self, frame, frame_num, team_colors=None):
        """
        Draw recent progressive passes as arrows on the frame,
        plus a stats panel.
        Returns annotated frame.
        """
        if team_colors is None:
            team_colors = {1: (0, 220, 100), 2: (0, 100, 220)}

        # Draw fading progressive pass arrows
        for (from_pos, to_pos, color, added_frame) in self._draw_buffer:
            age   = frame_num - added_frame
            if age > self.FADE_FRAMES:
                continue
            alpha = max(0.1, 1.0 - age / self.FADE_FRAMES)
            thickness = max(1, int(3 * alpha))

            overlay = frame.copy()
            cv2.arrowedLine(overlay, from_pos, to_pos, color,
                            thickness, tipLength=0.15)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Clean up old entries
        self._draw_buffer = [
            e for e in self._draw_buffer
            if frame_num - e[3] <= self.FADE_FRAMES
        ]

        # Stats panel — bottom right
        h, w = frame.shape[:2]
        px, py = w - 260, h - 130
        cv2.rectangle(frame, (px - 5, py), (w - 5, py + 120),
                      (30, 30, 30), -1)
        cv2.rectangle(frame, (px - 5, py), (w - 5, py + 120),
                      (100, 100, 100), 1)

        for i, team_id in enumerate([1, 2]):
            color  = team_colors.get(team_id, (200, 200, 200))
            prog   = self._pass_counts[team_id]
            total  = max(1, self._total_passes[team_id])
            rate   = prog / total * 100

            top_passer = self.get_top_progressive_passer(team_id)
            top_str = f"Top: #{top_passer[0][0]} ({top_passer[0][1]})" if top_passer else "Top: N/A"

            y = py + 20 + i * 55
            cv2.putText(frame, f"Team {team_id} Progressive",
                        (px, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Passes: {prog}  Rate: {rate:.0f}%",
                        (px, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, top_str,
                        (px, y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        return frame

    def get_top_progressive_passer(self, team_id, top_n=1):
        """Returns [(player_id, count)] for the top progressive passer(s)."""
        team_passers = {
            pid: cnt
            for pid, cnt in self._player_prog_passes.items()
            if cnt > 0
        }
        sorted_passers = sorted(team_passers.items(),
                                key=lambda x: x[1], reverse=True)
        return sorted_passers[:top_n]

    def get_summary(self):
        """Returns progressive pass stats for both teams."""
        summary = {}
        for team_id in [1, 2]:
            prog  = self._pass_counts[team_id]
            total = max(1, self._total_passes[team_id])
            summary[f"team_{team_id}"] = {
                "progressive_passes": prog,
                "total_passes":       self._total_passes[team_id],
                "prog_pass_rate_pct": round(prog / total * 100, 1),
                "top_passer":         self.get_top_progressive_passer(team_id),
            }
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_progressive(self, from_pos, to_pos, team_id):
        """
        Returns True if the pass is progressive for the given team.
        """
        goal_x = self.frame_width if self.attack_dir[team_id] == "right" else 0

        dist_before = abs(from_pos[0] - goal_x)
        dist_after  = abs(to_pos[0]   - goal_x)

        advance_px = dist_before - dist_after   # positive = moved toward goal

        if advance_px < self.MIN_ADVANCE_PX:
            return False

        if dist_before == 0:
            return False

        return (advance_px / dist_before) >= self.PROGRESS_RATIO