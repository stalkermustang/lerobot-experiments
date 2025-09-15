"""Record helper that toggles torque when teleoperating without a policy."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy
from lerobot.record import RecordConfig
from lerobot.record import record_loop as base_record_loop
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import _init_rerun


def record_loop(
    robot,
    events,
    fps: int,
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy=None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    should_disable = policy is None and isinstance(teleop, Teleoperator)
    if should_disable:
        teleop.bus.disable_torque()
    try:
        return base_record_loop(
            robot=robot,
            events=events,
            fps=fps,
            dataset=dataset,
            teleop=teleop,
            policy=policy,
            control_time_s=control_time_s,
            single_task=single_task,
            display_data=display_data,
        )
    finally:
        if should_disable:
            teleop.bus.enable_torque()


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = (
        make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    )

    action_features = hw_to_dataset_features(
        robot.action_features, "action", cfg.dataset.video
    )
    obs_features = hw_to_dataset_features(
        robot.observation_features, "observation", cfg.dataset.video
    )
    dataset_features = {**action_features, **obs_features}

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(
            dataset, robot, cfg.dataset.fps, dataset_features
        )
    else:
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
            * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    policy = (
        None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    )

    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while (
            recorded_episodes < cfg.dataset.num_episodes
            and not events["stop_recording"]
        ):
            log_say("Recording", cfg.play_sounds, blocking=True)
            log_say(f"episode {dataset.num_episodes}", cfg.play_sounds)
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop=teleop,
                policy=policy,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

            if not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1)
                or events["rerecord_episode"]
            ):
                log_say("Reset the environment", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop=teleop,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    record()


if __name__ == "__main__":
    main()
