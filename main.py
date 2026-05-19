from __future__ import annotations

from experiment_advisor import complete_trial, get_next_trial, initialize


def main() -> None:
    design = initialize()
    print("DOE design:")
    print(design)
    for _ in range(8):
        trial = get_next_trial()
        complete_trial(trial["trial_index"], {"yield": 80.0 + trial["trial_index"]})
    print("Next bayes recommendation:")
    print(get_next_trial())


if __name__ == "__main__":
    main()
