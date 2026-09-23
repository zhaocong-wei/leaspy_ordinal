"""
Ordinal Model
=============

This example illustrates the main steps for using the ``OrdinalModel``:

1. loading and inspecting ordinal longitudinal data;
2. fitting the population model;
3. estimating individual parameters;
"""

# %%
# Load the example longitudinal dataset. Each feature is measured on an ordered discrete scale.

from pathlib import Path

import leaspy.datasets as _leaspy_datasets
from leaspy.io.data import Data

data_path = Path(_leaspy_datasets.__file__).parent / "data" / "ordinal_example.csv"
data = Data.from_csv_file(data_path)

print(data.to_dataframe().head())

# %%
# Show how often each ordinal score occurs for each item.
# Rows are ordinal scores, columns are items (Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8), and each cell
# gives the number of observations with that score for that item.
# Items may have different ranges of scores.

import pandas as pd

raw_df = data.to_dataframe()
features = ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8"]

level_counts = pd.DataFrame({
    item: raw_df[item].value_counts(dropna=False)
    for item in features
}).fillna(0).astype(int)

level_counts.columns.name = "Item"
level_counts.index.name = "Ordinal score"

print(level_counts.sort_index().to_string())

# %%
# Fit the population model to all patients' longitudinal observations.
# This estimates the shared progression pattern before individual parameters are estimated in the next step.
#
# ``source_dimension`` sets the number of latent sources used to capture variation across items.
# A practical starting point is roughly the square root of the number of items. For three items, we choose two sources.

from leaspy.models import OrdinalModel

model = OrdinalModel(name="ordinal", source_dimension=2)
model.fit(
    data,
    "mcmc_saem",
    seed=42,
    n_iter=100,
    progress_bar=False,
)


# %%
# Inspect the fitted population parameters.
#
# As in the multivariate logistic model, the fitted population parameters
# include ``log_g_mean``, ``log_v0_mean``, ``tau_mean``, and ``betas_mean``.
# These describe the population trajectories and the effects of latent sources.

import numpy as np

for name in ("log_g_mean", "log_v0_mean", "tau_mean", "betas_mean"):
    print(f"{name}:")
    print(model.parameters[name])
    print()

# The ordinal model also estimates ``log_deltas_mean``. Its values are stored
# on a logarithmic scale. Below, we apply ``exp`` to show the ordinal delays
# on their original scale.
#
# Each row of the table represents an item. ``delta_h`` is the spacing between
# the transitions into levels h-1 and h: for example, ``delta_2`` is the
# spacing between the transitions into levels 1 and 2. ``delta_1`` is fixed
# to zero and is therefore not included in the table.
#
# Items can have different maximum levels. Positions that do not correspond
# to a transition may contain ``inf`` in ``log_deltas_mean``; the model masks
# these positions during its calculations. They appear as blank cells below.

log_deltas = model.parameters["log_deltas_mean"].detach().cpu().numpy()

deltas = np.full_like(log_deltas, np.nan)
valid = np.isfinite(log_deltas)
deltas[valid] = np.exp(log_deltas[valid])

delta_table = pd.DataFrame(
    deltas,
    index=pd.Index(features, name="Item"),
    columns=[
        f"delta_{h}"
        for h in range(2, log_deltas.shape[1] + 2)
    ],
)

print(delta_table.round(3).to_string(na_rep="—"))


# %%
