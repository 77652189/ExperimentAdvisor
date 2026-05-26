from __future__ import annotations

from pathlib import Path

from experiment_advisor.ingestion import build_final_dataset
from experiment_advisor.optimizer import BOEngine
from experiment_advisor.report import generate_recommendation_report
from experiment_advisor.space import build_search_space


def main() -> None:
    input_path = Path("data/final/fermentation_modeling_dataset.csv")
    if not input_path.exists():
        raise SystemExit("请先将历史数据放入 data/excel 或 data/csv，并调用 build_final_dataset() 生成 final 数据集。")

    df = build_final_dataset(input_path, output_path=input_path)
    engine = BOEngine(search_space=build_search_space())
    engine.cold_start(df)
    recommendations = engine.recommend(n=1)
    print(generate_recommendation_report(recommendations))


if __name__ == "__main__":
    main()
