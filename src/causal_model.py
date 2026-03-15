import pandas as pd
import networkx as nx
from dowhy import CausalModel
import warnings
warnings.filterwarnings('ignore')

data_path = r"C:\Users\Najib\Documents\Najib's Projects\Causal AI decision engine\bank-full-cleaned.csv"


def build_causal_model():

    df = pd.read_csv(data_path)

    # quick look at the raw numbers before any causal work
    cellular_rate    = df[df['treatment'] == 1]['outcome'].mean()
    noncellular_rate = df[df['treatment'] == 0]['outcome'].mean()
    raw_diff         = cellular_rate - noncellular_rate

    print(f"Cellular subscription rate    : {cellular_rate:.2%}")
    print(f"Non-cellular subscription rate: {noncellular_rate:.2%}")
    print(f"Raw difference (biased)       : {raw_diff:.2%}")
    print("Note: this gap is inflated by demographics — DoWhy will correct it\n")

    # build the causal graph as a networkx DiGraph
    # using networkx directly avoids the pydot/graphviz compatibility issues
    graph = nx.DiGraph()

    # confounders are pre-existing customer traits that affect both
    # who gets called via cellular AND whether they subscribe
    confounders = ['age', 'job', 'education', 'marital', 'balance', 'housing', 'loan', 'default']
    for node in confounders:
        graph.add_edge(node, 'treatment')
        graph.add_edge(node, 'outcome')

    # previous campaign history affects outcome but not contact method
    graph.add_edge('poutcome', 'outcome')
    graph.add_edge('was_previously_contacted', 'outcome')

    # the main relationship we're testing
    graph.add_edge('treatment', 'outcome')

    # note: duration, campaign, month, day are excluded because they are
    # post-treatment variables — they happen during or after the call,
    # so including them would corrupt the ATE estimate

    model = CausalModel(
        data      = df,
        treatment = 'treatment',
        outcome   = 'outcome',
        graph     = graph
    )

    # identify how to isolate the causal effect — DoWhy uses the backdoor criterion
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)

    # propensity score stratification groups customers by their likelihood
    # of being called via cellular — within each group they're comparable,
    # so the remaining difference in subscription rates is genuinely causal
    estimate = model.estimate_effect(
        identified_estimand,
        method_name  = "backdoor.propensity_score_stratification",
        target_units = "ate"
    )

    ate = estimate.value

    print(f"\nAverage Treatment Effect (ATE) : {ate:.4f}")
    print(f"Raw difference (before)        : {raw_diff:.2%}")
    print(f"True causal effect (after)     : {ate:.2%}")
    print(f"Selection bias removed         : {raw_diff - ate:.2%}")

    return model, identified_estimand, estimate, raw_diff


if __name__ == "__main__":
    model, identified_estimand, estimate, raw_diff = build_causal_model()