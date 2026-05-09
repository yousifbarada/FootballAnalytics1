import numpy as np
from sklearn.cluster import KMeans
import cv2


class FormationDetector:
    """
    Detects the tactical formation of each team (e.g., 4-3-3, 4-4-2)
    by clustering player positions into defensive, midfield, and attacking lines.
    """

    def __init__(self, n_lines=3):
        """
        Args:
            n_lines: Number of outfield lines to detect (default 3: defense/mid/attack).
                     Goalkeeper is always separated first.
        """
        self.n_lines = n_lines
        self.formation_history = {1: [], 2: []}   # team_id -> list of formations per frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_formation(self, tracks, frame_num, team_id):
        """
        Detect the formation for a given team at a specific frame.

        Args:
            tracks:     dict returned by the Tracker – tracks['players'][frame_num]
            frame_num:  int
            team_id:    1 or 2

        Returns:
            formation_str:  e.g. "4-3-3"
            line_groups:    list of player-id lists per line (GK excluded)
            gk_id:          track id of the detected goalkeeper (or None)
        """
        players_list = tracks.get("players", [{}])
        if frame_num >= len(players_list):
            return "Unknown", [], None
        players_in_frame = players_list[frame_num]

        # Collect positions for the requested team
        team_positions = {}
        for track_id, player_info in players_in_frame.items():
            if player_info.get("team") == team_id:
                bbox = player_info.get("bbox")
                if bbox is None:
                    continue
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                team_positions[track_id] = (cx, cy)

        if len(team_positions) < 4:
            return "Unknown", [], None

        formation_str, line_groups, gk_id = self._compute_formation(team_positions)

        self.formation_history[team_id].append(formation_str)
        return formation_str, line_groups, gk_id

    def get_dominant_formation(self, team_id):
        """Return the most frequently detected formation for a team."""
        history = self.formation_history.get(team_id, [])
        if not history:
            return "Unknown"
        return max(set(history), key=history.count)

    def draw_formation(self, frame, tracks, frame_num, team_id,
                       team_color=(0, 255, 0), text_color=(255, 255, 255)):
        """
        Overlay the current formation string and player lines on the frame.

        Returns the annotated frame.
        """
        formation_str, line_groups, gk_id = self.detect_formation(
            tracks, frame_num, team_id
        )

        players_list = tracks.get("players", [{}])
        if frame_num >= len(players_list):
            return frame
        players_in_frame = players_list[frame_num]

        # Draw lines connecting players in the same tactical line
        for line in line_groups:
            positions = []
            for track_id in line:
                if track_id in players_in_frame:
                    bbox = players_in_frame[track_id]["bbox"]
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    positions.append((cx, cy))

            # Sort by x so the connecting line looks natural
            positions.sort(key=lambda p: p[0])
            for i in range(len(positions) - 1):
                cv2.line(frame, positions[i], positions[i + 1], team_color, 2)

        # Draw formation label
        label_x = 20 if team_id == 1 else frame.shape[1] - 200
        label_y = 50 if team_id == 1 else 50
        cv2.putText(
            frame,
            f"Team {team_id}: {formation_str}",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            team_color,
            2,
            cv2.LINE_AA,
        )

        return frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_formation(self, team_positions):
        """
        Core logic:
        1. Separate the goalkeeper (player with extreme y-position).
        2. Cluster remaining players into self.n_lines groups by y-coordinate.
        3. Sort groups back → front and count players per group.

        Returns (formation_str, line_groups, gk_id)
        """
        track_ids = list(team_positions.keys())
        coords = np.array([team_positions[t] for t in track_ids])  # (N, 2)

        # --- Step 1: Isolate goalkeeper ---
        # GK is the player with the extreme y value (top or bottom of frame)
        y_values = coords[:, 1]
        gk_idx = int(np.argmax(y_values))   # bottom of frame = high y = defending team
        gk_id = track_ids[gk_idx]

        outfield_mask = np.ones(len(track_ids), dtype=bool)
        outfield_mask[gk_idx] = False

        outfield_ids = [track_ids[i] for i in range(len(track_ids)) if outfield_mask[i]]
        outfield_coords = coords[outfield_mask]

        if len(outfield_ids) < self.n_lines:
            return "Unknown", [], gk_id

        # --- Step 2: Cluster by y-coordinate ---
        n_clusters = min(self.n_lines, len(outfield_ids))
        y_only = outfield_coords[:, 1].reshape(-1, 1)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(y_only)

        # --- Step 3: Sort clusters back → front (descending y = deepest line first) ---
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_cluster_indices = np.argsort(cluster_centers)[::-1]  # high-y first

        line_groups = []
        line_counts = []
        for cluster_idx in sorted_cluster_indices:
            members = [
                outfield_ids[i]
                for i, lbl in enumerate(labels)
                if lbl == cluster_idx
            ]
            line_groups.append(members)
            line_counts.append(len(members))

        formation_str = "-".join(str(c) for c in line_counts)
        return formation_str, line_groups, gk_id