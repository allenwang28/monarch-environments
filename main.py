"""Sample code showing a few ways to track and handle episode execution time.

A few ways that straggler rollouts have been handled:
1. Timeout based truncation - stop when the episode has run for a certain amount of time
2. Step based truncation - stop when the episode has run for a certain number of steps

"""
import random
import time
import math
import asyncio
from monarch.actor import Actor, endpoint, proc_mesh, current_rank, current_size


class ToyCollector(Actor):
    def __init__(
        self,
        low_steps: int,
        high_steps: int,
        step_time: float,
        max_steps: int | None = None,
        max_time: int | None = None):

        self.low_steps = low_steps
        self.high_steps = high_steps
        self.step_time = step_time
        self._rank = current_rank().rank
        self._size = math.prod(current_size().values())
        self._max_steps = max_steps
        self._max_time = max_time

    def log(self, msg: str):
        print(f"[{self._rank}/{self._size}] {msg}")

    @endpoint
    async def run_episode(self) -> int:
        # Randomly create a set of steps and run them
        num_steps = random.randint(self.low_steps, self.high_steps)
        self.log(f"Running episode with {num_steps} steps")

        start = time.time()

        for step in range(num_steps):
            # Simulate step execution
            self.log(f"Executing step {step + 1}/{num_steps}")
            time.sleep(self.step_time)

            if self._max_steps is not None and step >= self._max_steps:
                self.log(f"Forcing episode completion at {step + 1} steps")
                return step + 1

            if self._max_time is not None and time.time() - start >= self._max_time:
                self.log(f"Forcing episode completion at {step + 1} steps as {self._max_time} seconds have passed")
                return step + 1


        self.log(f"Episode completed with {num_steps} steps")
        return num_steps


async def main():
    # note - while we say gpus here, this is actually the number of processes...
    p = await proc_mesh(gpus=4)
    env = await p.spawn("env", ToyCollector, 1, 20, 0.2, max_time=2.)
    await env.run_episode.call()


if __name__ == "__main__":
    asyncio.run(main())
