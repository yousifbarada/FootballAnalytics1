from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from formation_detector import FormationDetector
from PressIntensityAnalyzer import PressIntensityAnalyzer
from Bildupanalyzer import BuildUpAnalyzer
from ProgressPassD import ProgressivePassDetector


def main():
    # ── 1. Read Video ─────────────────────────────────────────────────────────
    video_frames = read_video(r"/teamspace/studios/this_studio/input_videos/Video Project 17 2.mp4")
    frame_h, frame_w = video_frames[0].shape[:2]

    # ── 2. Tracking ───────────────────────────────────────────────────────────
    tracker = Tracker(r'/teamspace/studios/this_studio/models/players.pt')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=r'/teamspace/studios/this_studio/stubs/vedio.pkl'
    )

    if "referee" in tracks and "referees" not in tracks:
        tracks["referees"] = tracks.pop("referee")

    # ── 3. Add positions ──────────────────────────────────────────────────────
    tracker.add_position_to_tracks(tracks)

    # ── 4. Camera movement ────────────────────────────────────────────────────
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=False,
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # ── 5. View Transformer ───────────────────────────────────────────────────
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # ── 6. Interpolate Ball ───────────────────────────────────────────────────
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # ── 7. Speed and Distance ─────────────────────────────────────────────────
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # ── 8. Team Assignment ────────────────────────────────────────────────────
    team_assigner = TeamAssigner()

    initial_frame_idx = 0
    for frame_num, player_dict in enumerate(tracks['players']):
        if len(player_dict) >= 5:
            initial_frame_idx = frame_num
            break

    team_assigner.assign_team_color(
        video_frames[initial_frame_idx],
        tracks['players'][initial_frame_idx]
    )

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track['bbox'], player_id
            )
            if team is not None:
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = \
                    team_assigner.team_colors[team]

    # ── 9. Ball Acquisition ───────────────────────────────────────────────────
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        if 1 not in tracks['ball'][frame_num]:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            continue

        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player].get('team', 0)
            )
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # ── 10. Draw Original Annotations ────────────────────────────────────────
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # ── 11. Init NEW features ─────────────────────────────────────────────────
    press_analyzer     = PressIntensityAnalyzer()
    buildup_analyzer   = BuildUpAnalyzer(frame_width=frame_w)
    # prog_pass_detector = ProgressivePassDetector(frame_width=frame_w)
    formation_detector = FormationDetector(n_lines=3)

    team_colors = {
        1: tuple(int(c) for c in team_assigner.team_colors.get(1, (0, 220, 100))),
        2: tuple(int(c) for c in team_assigner.team_colors.get(2, (0, 100, 220))),
    }

    # ── 12. Draw NEW Features (update + draw في نفس الـ loop) ─────────────────
    print("[INFO] Rendering new features ...")
    for frame_num, frame in enumerate(output_video_frames):

        # ✅ UPDATE أولاً — عشان الـ draw يشوف القيم الجديدة
        press_analyzer.update(tracks, frame_num)
        buildup_analyzer.update(tracks, frame_num)
        # prog_pass_detector.update(tracks, frame_num)

        # ✅ DRAW بعدها مباشرةً
        frame = press_analyzer.draw(frame, team_colors=team_colors)
        frame = buildup_analyzer.draw(frame, team_colors=team_colors)
        # frame = prog_pass_detector.draw(frame, frame_num, team_colors=team_colors)
        frame = formation_detector.draw_formation(
            frame, tracks, frame_num, team_id=1, team_color=(0, 255, 120)
        )
        frame = formation_detector.draw_formation(
            frame, tracks, frame_num, team_id=2, team_color=(0, 120, 255)
        )

        output_video_frames[frame_num] = frame

    # ── 13. Save Output ───────────────────────────────────────────────────────
    save_video(output_video_frames, 'output_videos/output_video2.avi')

    # ── 14. Print Summary ─────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("TACTICAL SUMMARY")
    print("="*50)

    press_summary = press_analyzer.get_summary()
    print("\nPress Intensity:")
    for team, stats in press_summary.items():
        print(f"  {team}: {stats['intensity_pct']}%")

    buildup_summary = buildup_analyzer.get_summary()
    print("\nBuild-Up Analysis:")
    for team, stats in buildup_summary.items():
        print(f"  {team}: zones D={stats['zone_pct']['D']}% "
              f"M={stats['zone_pct']['M']}% A={stats['zone_pct']['A']}% "
              f"| Success Rate: {stats['success_rate_pct']}%")

    # prog_summary = prog_pass_detector.get_summary()
    # print("\nProgressive Passes:")
    # for team, stats in prog_summary.items():
    #     print(f"  {team}: {stats['progressive_passes']} prog passes "
    #           f"({stats['prog_pass_rate_pct']}% of all passes) "
    #           f"| Top passer: {stats['top_passer']}")

    print(f"\n[Formation] Team 1: {formation_detector.get_dominant_formation(1)}")
    print(f"[Formation] Team 2: {formation_detector.get_dominant_formation(2)}")
    print("\n✅ Done! Output saved to output_videos/output_video2.avi")


if __name__ == '__main__':
    main()