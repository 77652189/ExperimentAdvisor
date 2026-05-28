import pandas as pd
from experiment_advisor.recommendation.service import compare_recommenders

df = pd.read_csv("data/final/run_level_modeling_dataset.csv")
result = compare_recommenders(df, top_k=3, seed=0, method="ei")
print("method_key:", result["model_info"]["primary_method"])
print("n_recs:", len(result["selected_recommendations"]))
for r in result["selected_recommendations"]:
    print(
        f"  Rank {r['rank']}: yield={r['predicted_yield']:.1f}"
        f"  uncertainty={r['model_uncertainty']:.1f}"
        f"  method={r['method']}"
    )
print("type labels:", [
    item.get("recommendation_type")
    for item in result["strategy_quality"]["per_recommendation"]
])
