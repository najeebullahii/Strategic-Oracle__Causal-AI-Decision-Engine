import pandas as pd
import networkx as nx
from dowhy import CausalModel
import warnings
warnings.filterwarnings('ignore')

data_path    = r"C:\Users\Najib\Documents\Najib's Projects\Causal AI decision engine\bank-full-cleaned.csv"
original_ate = 0.0681


def build_model(df):
    # exact same graph as causal_model.py — must be identical for tests to be valid
    graph = nx.DiGraph()

    confounders = ['age', 'job', 'education', 'marital', 'balance', 'housing', 'loan', 'default']
    for node in confounders:
        graph.add_edge(node, 'treatment')
        graph.add_edge(node, 'outcome')

    graph.add_edge('poutcome', 'outcome')
    graph.add_edge('was_previously_contacted', 'outcome')
    graph.add_edge('treatment', 'outcome')

    return CausalModel(data=df, treatment='treatment', outcome='outcome', graph=graph)


def check_result(new_value, expect_zero=False):
    # helper to print a quick pass/fail for each test
    if expect_zero:
        passed = abs(new_value) < 0.01
        print(f"  new effect : {new_value:.4f}  (expected ~0)")
        print(f"  result     : {'PASSED' if passed else 'FAILED'}")
    else:
        shift  = abs(new_value - original_ate)
        passed = shift < original_ate * 0.20
        print(f"  new effect : {new_value:.4f}  (expected ~{original_ate:.4f})")
        print(f"  shift      : {shift:.4f}  ({(shift/original_ate)*100:.1f}%)")
        print(f"  result     : {'PASSED' if passed else 'FAILED'}")


def run_refutation_tests():

    df    = pd.read_csv(data_path)
    model = build_model(df)

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified_estimand,
        method_name  = "backdoor.propensity_score_stratification",
        target_units = "ate"
    )

    print(f"Original ATE: {estimate.value:.4f}")
    print("Running 4 refutation tests — this takes a few minutes\n")

    # test 1: replace real treatment with random noise
    # if the model is genuine, the effect should collapse to near zero
    print("Test 1 — Placebo Treatment")
    placebo = model.refute_estimate(
        identified_estimand, estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        num_simulations=20
    )
    print(placebo)
    check_result(placebo.new_effect, expect_zero=True)

    # test 2: add a completely fake random confounder
    # ATE should barely move if the causal structure is solid
    print("\nTest 2 — Random Common Cause")
    random_cause = model.refute_estimate(
        identified_estimand, estimate,
        method_name="random_common_cause",
        num_simulations=20
    )
    print(random_cause)
    check_result(random_cause.new_effect)

    # test 3: drop 10% of the data at random and rerun
    # a stable finding shouldn't depend on any particular slice of rows
    print("\nTest 3 — Data Subset")
    subset = model.refute_estimate(
        identified_estimand, estimate,
        method_name="data_subset_refuter",
        subset_fraction=0.9,
        num_simulations=20
    )
    print(subset)
    check_result(subset.new_effect)

    # test 4: resample the data 20 times and check consistency
    # if the ATE stays stable across resamples, the finding is reliable
    print("\nTest 4 — Bootstrap")
    bootstrap = model.refute_estimate(
        identified_estimand, estimate,
        method_name="bootstrap_refuter",
        num_simulations=20
    )
    print(bootstrap)
    check_result(bootstrap.new_effect)

    print("\nSummary")
    print(f"  original ATE  : {original_ate:.4f}")
    print(f"  placebo       : {placebo.new_effect:.4f}  (should be ~0)")
    print(f"  random cause  : {random_cause.new_effect:.4f}  (should be ~{original_ate:.4f})")
    print(f"  data subset   : {subset.new_effect:.4f}  (should be ~{original_ate:.4f})")
    print(f"  bootstrap     : {bootstrap.new_effect:.4f}  (should be ~{original_ate:.4f})")
    print("\np-value > 0.05 on all four tests = model is statistically robust")


if __name__ == "__main__":
    run_refutation_tests()