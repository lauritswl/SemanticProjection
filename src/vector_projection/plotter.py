# Import necessary libraries
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Plotter:
   """
   Class to handle plotting of projection results and calculating correlations.
         Methods:
         - marginal_scatterplot: Plots projections against labels and calculates Spearman correlation.
   """   
   @staticmethod
   def marginal_scatterplot(projections, labels, xlabel='Fiction4 Subspace Projection', y_label='Human Gold Standard', title='Scatterplot with Correlation (MPNET Standardized)', save_path=None):
       """
         Plots the projections against the labels and calculates Spearman correlation.
            Args:
                projections: pd.Series or np.array, projected values (e.g. ProjectionAnalyzer.projected_in_1D from core/projection.py)
                labels: pd.Series or np.array, true labels (e.g. Loader.test_labels from core/loader.py)
                xlabel: str, label for the x-axis
                y_label: str, label for the y-axis
                title: str, title of the plot
                save_path: str or None, if provided, saves the plot to this path.
            Returns:
                g: seaborn JointGrid object containing the plot
         """
       # 1. Standardize projections
       scaler = StandardScaler()
       scaled = scaler.fit_transform(projections.values.reshape(-1, 1)).flatten()

       # 2. Calculate Spearman correlation
       corr, _ = spearmanr(scaled, labels)

       # 3. Create the plot
       g = sns.jointplot(x=scaled, y=labels, kind="scatter", marginal_kws=dict(fill=True),
                        alpha=0.5, height=8)  
       g.ax_joint.set(xlabel=xlabel, ylabel=y_label)
       g.ax_joint.set_title(title, pad=90)
       g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr:.2f}', transform=g.ax_joint.transAxes,
                       fontsize=12, color='blue')
       g.figure.subplots_adjust(top=0.98, bottom=0.15)
       if save_path:
          plt.savefig(save_path, bbox_inches="tight")
       
   @staticmethod
   def category_correlation_table(df, category_col="category", label_col="label", pred_col="prediction",dataset = None, save_path=None):
        """
        Compute correlation between label and prediction for each category 
        and show it as a matplotlib table.
        """
        results = []
        for cat, group in df.groupby(category_col):
            if len(group) > 1:  # need at least 2 points
                r, _ = spearmanr(group[label_col], group[pred_col])
            else:
                r = float("nan")
            results.append({"Category": cat, "N": len(group), "Correlation": r})

        table_df = pd.DataFrame(results).sort_values("Correlation", ascending=False)

        # Plot table
        fig, ax = plt.subplots(figsize=(6, 0.5*len(table_df)+1))
        ax.axis("off")
        mpl_table = ax.table(
            cellText=table_df.round(3).values,
            colLabels=table_df.columns,
            cellLoc="center",
            loc="center"
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(10)
        mpl_table.scale(1, 1.3)
        plt.title(f"Category-wise Correlation between Label and Prediction, \n Dataset: {dataset}", pad=5)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")




        