from dataclasses import dataclass

from bicm_metrics import SampleBatch, generate_sample_batch


@dataclass(frozen=True)
class MultiUserSampleAverage:
    batches: tuple[tuple[SampleBatch, ...], ...]

    def __post_init__(self) -> None:
        """?????????????????"""
        if not self.batches:
            raise ValueError("At least one sample batch is required.")
        num_users = len(self.batches[0])
        for batch_group in self.batches:
            if len(batch_group) != num_users:
                raise ValueError("All batch groups must have the same number of users.")

    @property
    def num_repeats(self) -> int:
        """?? num repeats ???"""
        return len(self.batches)

    @property
    def num_users(self) -> int:
        """?? num users ???"""
        return len(self.batches[0])


def build_multiuser_sample_average(
    env,
    bits_per_symbol: int,
    num_samples: int,
    num_repeats: int,
    base_seed: int,
    labeling: str = "gray_standard",
) -> MultiUserSampleAverage:
    """Build shared Monte Carlo batches for all users."""
    batches = []
    for repeat_idx in range(num_repeats):
        user_batches = []
        for user_idx in range(env.num_users):
            seed = base_seed + repeat_idx * env.num_users + user_idx
            user_batches.append(
                generate_sample_batch(
                    bits_per_symbol=bits_per_symbol,
                    num_streams=env.num_streams_per_user,
                    num_samples=num_samples,
                    seed=seed,
                    labeling=labeling,
                )
            )
        batches.append(tuple(user_batches))
    return MultiUserSampleAverage(batches=tuple(batches))


# Backward-compatible aliases for the older AO implementation.
MultiUserSICSampleAverage = MultiUserSampleAverage


def build_multiuser_sic_sample_average(
    env,
    bits_per_symbol: int,
    num_samples: int,
    num_repeats: int,
    base_seed: int,
    labeling: str = "gray_standard",
) -> MultiUserSICSampleAverage:
    return build_multiuser_sample_average(
        env=env,
        bits_per_symbol=bits_per_symbol,
        num_samples=num_samples,
        num_repeats=num_repeats,
        base_seed=base_seed,
        labeling=labeling,
    )
