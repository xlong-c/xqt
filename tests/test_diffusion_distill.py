import pytest

from xqt.diffusion_distill import build_step_schedule, downsample_timesteps


def test_downsample_timesteps_selects_even_teacher_steps() -> None:
    assert downsample_timesteps(20, 4) == [19, 12, 6, 0]
    assert downsample_timesteps(8, 1) == [7]


def test_build_step_schedule_pairs_teacher_and_student_steps() -> None:
    schedule = build_step_schedule(10, 3)

    assert [(pair.teacher_step, pair.student_step) for pair in schedule] == [
        (9, 0),
        (4, 1),
        (0, 2),
    ]


@pytest.mark.parametrize(
    ("teacher_steps", "student_steps", "message"),
    [
        (0, 1, "teacher_steps"),
        (4, 0, "student_steps"),
        (2, 4, "less than or equal"),
    ],
)
def test_downsample_timesteps_rejects_invalid_steps(
    teacher_steps: int,
    student_steps: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        downsample_timesteps(teacher_steps, student_steps)
