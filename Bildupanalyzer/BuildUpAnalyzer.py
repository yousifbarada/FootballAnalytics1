import numpy as np
import cv2
from collections import defaultdict


class BuildUpAnalyzer:
    """
    Analyses how a team builds play from the back.

    Divides the pitch into 3 vertical zones (defensive / middle / attacking)
    and tracks ball progression through zones per team.

    Metrics:
        - Zone distribution: % of time ball is in each zone per team
        - Build-up sequences: consecutive zone progressions (D→M→A)
        - Build-up success rate: sequences that reach the attacking third
    """

    # Pitch zone thresholds as fraction of frame width
    DEFENSIVE_ZONE  = 0.33   # 0 – 33%
    ATTACKING_ZONE  = 0.66   # 66 – 100%

    def __init__(self, frame_width):
        self.frame_width = frame_width

        # team_id -> list of zones visited while team had possession
        self._possession_zones = defaultdict(list)

        # team_id -> completed sequences [[z1,z2,...], ...]
        self._sequences        = defaultdict(list)
        self._current_seq      = {1: [], 2: []}
        self._last_team        = None

        # zone counts per team: {team_id: {zone: count}}
        self._zone_counts = defaultdict(lambda: {"D": 0, "M": 0, "A": 0})

        # successful build-ups (D→M→A in order)
        self._successful_buildups = {1: 0, 2: 0}
        self._total_sequences     = {1: 0, 2: 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tracks, frame_num):
        """
        Call once per frame.

        Args:
            tracks:    Tracker dict
            frame_num: int
        """
        players_list = tracks.get("players", [])
        ball_list    = tracks.get("ball", [])

        # Find who has the ball
        possession_team = None
        ball_x          = None

        if frame_num < len(players_list):
            for _, info in players_list[frame_num].items():
                if info.get("has_ball"):
                    possession_team = info.get("team")
                    bbox = info.get("bbox", [])
                    if len(bbox) == 4:
                        ball_x = (bbox[0] + bbox[2]) / 2
                    break

        # Fallback to ball track
        if ball_x is None and frame_num < len(ball_list):
            bbox = ball_list[frame_num].get(1, {}).get("bbox", [])
            if len(bbox) == 4:
                ball_x = (bbox[0] + bbox[2]) / 2

        if ball_x is None or possession_team is None:
            return

        # Determine zone
        ratio = ball_x / self.frame_width
        if ratio < self.DEFENSIVE_ZONE:
            zone = "D"
        elif ratio < self.ATTACKING_ZONE:
            zone = "M"
        else:
            zone = "A"

        self._zone_counts[possession_team][zone] += 1

        # Build-up sequence tracking
        if possession_team != self._last_team:
            # Possession changed — close previous sequence
            if self._last_team is not None and self._current_seq[self._last_team]:
                self._close_sequence(self._last_team)
            self._current_seq[possession_team] = [zone]
        else:
            seq = self._current_seq[possession_team]
            if not seq or seq[-1] != zone:
                seq.append(zone)

        self._last_team = possession_team

    def _close_sequence(self, team_id):
        seq = self._current_seq[team_id]
        if len(seq) >= 2:
            self._sequences[team_id].append(seq[:])
            self._total_sequences[team_id] += 1
            # Successful = contains D→M→A in order
            if self._is_successful(seq):
                self._successful_buildups[team_id] += 1
        self._current_seq[team_id] = []

    def _is_successful(self, seq):
        """True if sequence visits D, then M, then A in order."""
        try:
            d_idx = seq.index("D")
            m_idx = seq.index("M", d_idx)
            a_idx = seq.index("A", m_idx)
            return True
        except ValueError:
            return False

    def draw(self, frame, team_colors=None):
        """
        Draw zone distribution bars and build-up success rate on frame.
        Returns annotated frame.
        """
        if team_colors is None:
            team_colors = {1: (0, 220, 100), 2: (0, 100, 220)}

        h, w = frame.shape[:2]

        # Draw pitch zone lines (subtle)
        d_x = int(w * self.DEFENSIVE_ZONE)
        a_x = int(w * self.ATTACKING_ZONE)
        cv2.line(frame, (d_x, 0), (d_x, h), (180, 180, 180), 1)
        cv2.line(frame, (a_x, 0), (a_x, h), (180, 180, 180), 1)

        # Zone labels at top
        for label, x in [("DEF", 10), ("MID", d_x + 10), ("ATT", a_x + 10)]:
            cv2.putText(frame, label, (x, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Stats panel top-right
        panel_x = w - 260
        panel_y = 10
        cv2.rectangle(frame, (panel_x - 5, panel_y),
                      (w - 5, panel_y + 110), (30, 30, 30), -1)
        cv2.rectangle(frame, (panel_x - 5, panel_y),
                      (w - 5, panel_y + 110), (100, 100, 100), 1)

        for i, team_id in enumerate([1, 2]):
            color  = team_colors.get(team_id, (200, 200, 200))
            counts = self._zone_counts[team_id]
            total  = max(1, sum(counts.values()))

            d_pct = counts["D"] / total * 100
            m_pct = counts["M"] / total * 100
            a_pct = counts["A"] / total * 100

            success = self._successful_buildups[team_id]
            seqs    = max(1, self._total_sequences[team_id])
            success_rate = success / seqs * 100

            y = panel_y + 20 + i * 55
            cv2.putText(frame, f"Team {team_id} Build-Up",
                        (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1, cv2.LINE_AA)
            cv2.putText(frame,
                        f"D:{d_pct:.0f}% M:{m_pct:.0f}% A:{a_pct:.0f}%",
                        (panel_x, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame,
                        f"Success Rate: {success_rate:.0f}%  ({success}/{seqs-1})",
                        (panel_x, y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        return frame

    def get_summary(self):
        """Returns build-up stats for both teams."""
        summary = {}
        for team_id in [1, 2]:
            counts = self._zone_counts[team_id]
            total  = max(1, sum(counts.values()))
            seqs   = max(1, self._total_sequences[team_id])
            summary[f"team_{team_id}"] = {
                "zone_pct": {z: round(counts[z]/total*100, 1) for z in "DMA"},
                "successful_buildups": self._successful_buildups[team_id],
                "total_sequences": self._total_sequences[team_id],
                "success_rate_pct": round(
                    self._successful_buildups[team_id] / seqs * 100, 1
                ),
            }
        return summary