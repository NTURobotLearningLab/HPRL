from __future__ import annotations
from typing import Any, SupportsFloat

from gymnasium import Wrapper
from ..base.dsl_nodes import dsl_nodes
from ..base import BaseDSL
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from .dsl import MinigridDSL
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC
from minigrid.core.world_object import WorldObj, Door
from minigrid.core.grid import Grid
import os
from PIL import Image

from copy import deepcopy


class RewardWrapper(Wrapper):

    def __init__(self, env: MiniGridEnv):
        self.unwrapped: MiniGridEnv
        self.crushed: bool = False
        super().__init__(env)

    def _reward(self) -> float:
        """
        Overlap the step penalty
        """

        return 1

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, _ = self.env.step(action)
        if reward > 0:
            reward = self._reward()

        return _, reward, terminated, truncated, _


class ProgramWrapper(Wrapper):

    def __init__(
        self,
        env: MiniGridEnv,
        seed: int,
        crashable: bool = False,
        crash_penalty: float = -0.01,
        max_calls: int = 1000,
        record_history: bool = False,
    ):
        self.seed: int = seed
        self.reward: float = 0.0
        self.terminated: bool = False
        self.info = {}
        self.actions = {
            "left": self.left,
            "right": self.right,
            "forward": self.forward,
            "pickup": self.pickup,
            "drop": self.drop,
            "toggle": self.toggle,
        }
        self.bool_features = {
            "front_is_clear": self.front_is_clear,
            "front_object_type": self.front_object_type,
            "front_object_color": self.front_object_color,
            "is_carrying_object": self.is_carrying_object,
        }
        self.has_parameters = {
            "front_is_clear": False,
            "front_object_type": True,
            "front_object_color": True,
            "is_carrying_object": False,
        }
        # Maximum num of calls
        self.max_calls: int = max_calls
        self.num_calls: int = 0
        # Whether the env will terminate whenever perform invalid actions
        self.crashable: bool = crashable
        # Whenever perform invalid action, get penalty
        self.crash_penalty = crash_penalty
        self.crushed: bool = False
        # Whether to record the history the agent
        self.record_history = record_history
        self.history = []
        self.program_num = 0
        super().__init__(env)

    def __eq__(self, other):
        return self.hash() == other.hash()

    def reset(self) -> tuple[Any, dict[str, Any]]:
        self.reward = 0.0
        self.terminated = False
        self.info = {}
        self.num_calls = 0
        self.crushed = False
        self.history = []
        return super().reset(seed=self.seed)

    def evaluate_program(
        self,
        program: dsl_nodes.Program,
        record_video: bool = False,
        record_dir: str = None,
    ) -> float:
        self.reset()
        reward = 0.0
        self.program_num += 1
        if record_video:
            images = []
        for _ in program.run_generator(self):
            if record_video:
                images.append(self.get_frame())
            terminated, instant_reward = self.get_reward()
            reward += instant_reward
            if terminated:
                break

        if record_video:
            assert record_dir != None
            os.makedirs(record_dir, exist_ok=True)
            for i, image in enumerate(images):
                name = os.path.join(record_dir, f"{str(i)}.png")
                im = Image.fromarray(image)
                im.save(name)
        return reward

    def generate_action_history(
        self,
        action: str,
        unchanged: bool,
        prev_state: str,
        prev_agent_pos: list[int],
        prev_agent_dir: list[int],
        prev_grid: Grid,
    ) -> dict[str:str, str:str, str:str, str:str]:
        history = {
            "type": "action",
            "name": action,
            "feedback": None,
            "prev_state": prev_state,
            "cur_state": self.print_state(),
            "result": None,
        }
        if unchanged:
            history["feedback"] = "Nothing happened"
            history["result"] = (
                f"the returned reward is ` {str(self.reward)} ` and the returned done is ` {str(self.terminated)} `"
            )
            return history

        if action == "forward":
            agent_pos = self.agent_pos
            history["feedback"] = (
                f"The agent (pos=({prev_agent_pos[0]}, {prev_agent_pos[1]}) with direction ({prev_agent_dir[0]}, {prev_agent_dir[1]}) becomes an agent pos=({agent_pos[0]}, {agent_pos[1]}))"
            )
            history["result"] = (
                f"the returned reward is ` {str(self.reward)} ` and the returned done is ` {str(self.terminated)} `"
            )
            return history

        if action == "left" or action == "right":
            agent_dir = DIR_TO_VEC[self.agent_dir]
            history["feedback"] = (
                f"The agent (direction=({prev_agent_dir[0]}, {prev_agent_dir[1]}) at pos ({prev_agent_pos[0]}, {prev_agent_pos[1]}) becomes an agent (direction=({prev_agent_dir[0]}, {[prev_agent_dir[1]]}))"
            )
            history["result"] = (
                f"the returned reward is ` {str(self.reward)} ` and the returned done is ` {str(self.terminated)} `"
            )
            return history

        if action == "pickup":
            fwd_pos = self.unwrapped.front_pos
            fwd_cell = prev_grid.get(*fwd_pos)

            history["feedback"]

    def run_action(self, action: str):
        self.num_calls += 1
        old_hash = self.env.hash()
        self.actions[action]()
        if self.env.hash() == old_hash:
            if self.crashable:
                self.crush()
            else:
                self.reward += self.crash_penalty
        if self.num_calls >= self.max_calls:
            self.crushed = True

    def get_bool_feature(self, feature: str):
        self.num_calls += 1
        if self.num_calls >= self.max_calls:
            self.crushed = True
        if self.has_parameters[feature]:
            return self.bool_features[feature]
        else:
            return self.bool_features[
                feature
            ]()  # no parameters, directly call the function

    def get_reward(self):
        terminated = self.terminated
        reward = self.reward
        if terminated:
            self.crash()
        self.reward = 0.0

        return terminated, reward

    def crash(self):
        self.crushed = True

    def is_crashed(self):
        return self.crushed

    def left(self):
        _, reward, terminated, truncated, _ = self.env.step(Actions.left)
        self.reward = reward
        self.terminated = terminated

    def right(self):
        _, reward, terminated, truncated, _ = self.env.step(Actions.right)
        self.reward = reward
        self.terminated = terminated

    def forward(self):
        _, reward, terminated, truncated, _ = self.env.step(Actions.forward)
        self.reward = reward
        self.terminated = terminated

    def pickup(self):
        _, reward, terminated, truncated, _ = self.env.step(Actions.pickup)
        self.reward = reward
        self.terminated = terminated

    def drop(self):
        _, reward, terminated, truncated, _ = self.env.step(Actions.drop)
        self.reward = reward
        self.terminated = terminated

    def toggle(self):
        _, reward, terminated, truncated, _ = self.env.step(Actions.toggle)
        self.reward = reward
        self.terminated = terminated

    def front_is_clear(self) -> bool:
        fwd_pos = self.unwrapped.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        return fwd_cell is None or fwd_cell.can_overlap()

    def front_object_type(self, obj_name: str) -> bool:
        fwd_pos = self.unwrapped.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell is None:
            return False
        return fwd_cell.type == obj_name

    def front_object_color(self, color: str) -> bool:
        fwd_pos = self.unwrapped.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell is None:
            return False
        return fwd_cell.color == color

    def is_carrying_object(self) -> bool:
        return self.carrying is not None
