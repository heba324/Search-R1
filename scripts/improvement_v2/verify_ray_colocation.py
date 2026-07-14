"""Fail fast when the V2 actor wrapper is unsafe for verl Ray colocation."""

from scripts.improvement_v2.seeded_worker import install_seeded_rollout_patch
from verl.single_controller.base import Worker
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def main() -> None:
    import ray

    patched_worker = install_seeded_rollout_patch(ActorRolloutRefWorker)
    if patched_worker is not ActorRolloutRefWorker:
        raise RuntimeError("Seed installation must preserve the upstream worker class.")

    remote_worker = ray.remote(patched_worker)
    ray_base = remote_worker.__ray_actor_class__.__base__
    if ray_base is not Worker:
        raise RuntimeError(
            "V2 worker is unsafe for Ray colocation: "
            f"expected base {Worker}, found {ray_base}."
        )

    print("CEGR V2 Ray colocation compatibility passed.")


if __name__ == "__main__":
    main()
