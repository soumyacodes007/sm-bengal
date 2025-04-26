# SMART BENGAL HACKATHON DATA PRIVCY CHALLENGE 

**TEAM LEAD** = Soumyadeep Bhowmik



TLDR 

Key Metrics & Results:
Initial High+Medium Confidence Match Rate: 98.91%.
Matches potentially corrected by ML: 575.
Matches potentially corrected by Distance Refinement: 799.
Final Matched Records: matched_records_final.csv.
Key Finding: Protection primarily involved minor scaling (factor ~1.0) and low-variance Gaussian-like noise around zero difference for most variables.

## 1. Project Goal

The primary objective of this project was to evaluate the effectiveness of privacy protection techniques applied to a dataset containing individuals' income and expenditure information
# Adversarial Attack on Privacy: Record Linkage Analysis

## Project Overview

This project implements a sophisticated adversarial attack to assess the effectiveness of privacy protections applied to a dataset containing individual income and expenditure information. By attempting to link records in a "protected" dataset back to their counterparts in the "original" dataset, we evaluate the extent to which anonymity has been preserved.

The solution employs a multi-stage approach combining feature engineering, similarity matching, machine learning refinement, transformation/noise analysis, and privacy quantification (epsilon estimation) to achieve. We aimed to achieve this by attempting to re-identify records in the "protected" dataset by linking them back to the provided "original" dataset. Success in this linkage task highlights potential vulnerabilities in the anonymization method used.

## 2. Methodology: A Multi-Stage Approach

We employed a sophisticated, multi-stage strategy designed to maximize re-identification accuracy and provide deep insights into the nature of the data alterations:

1.  **Stable Fingerprinting:**
    *   Created unique profiles high linkage accuracy and provide deep insights into the vulnerabilities of the protection method.

**Challenge Goal:** Unmask private information, ("fingerprints") for each record focusing on features less sensitive to simple numerical alterations than raw values.
    *   Engine match protected rows to original rows, and analyze the privacy problems encountered.

## Methodology

Our solution follows a systematic pipeline:

1.  **Stable Fingerprinting:**
    *   **Goal:** Create robust representations of records resistant to simple data alterations.
    *   **Technered features included: expense-to-income ratios, spending ranks (highest to lowest expense), outlier flags for unusual spending,iques:** Calculated expense-to-income ratios, spending ranks (highest to lowest expense), outlier flags for unusual spending, and composite demographic features (e.g., age*dependents, occupation+city_tier).
    *   *Rationale and composite demographic features (e.g., `Age * Dependents`).
    *   **Rationale:** Relative values and patterns are often:* Patterns and relative values often persist better than absolute numbers under anonymization.

2.  **High-Recall Initial Matching more stable than absolute numbers under noise or scaling.

2.  **High-Recall Initial Matching:**
    *   **Goal:**
    *   Used `StandardScaler` and `cosine_similarity` (from `scikit-learn`) on the numerical fingerprints to calculate similarity between all protected-original pairs.
    *   Identified initial high-confidence matches (>97%:** Efficiently find the most likely matches and establish a baseline.
    *   **Techniques:** Applied Standard Scaling similarity) and top candidates for remaining records.
    *   *Rationale:* Efficiently finds obvious matches and establishes a strong to numerical fingerprints and calculated Cosine Similarity between all protected-original pairs. Identified unique high-confidence matches (>97% baseline.

3.  **ML-Powered Refinement:**
    *   Trained a `RandomForestClassifier` ( similarity) greedily and stored top candidates for others.
    *   **Rationale:** Cosine similarity effectively compares highfrom `scikit-learn`) to predict the probability of a true match for record pairs.
    *   Features-dimensional profiles; scaling ensures fair feature contribution. Provides a strong starting point.

3.  **ML-Powered included differences/ratios of core attributes, categorical matches, and the initial similarity score.
    *   The model was trained on high-confidence matches Refinement:**
    *   **Goal:** Improve accuracy by learning complex matching patterns missed by simple similarity.
    *   **Techn (positive examples) and randomly generated non-matches (negative examples).
    *   Applied the trained model to refine the matches for records not initially classified as high-confidence.
    *   *Rationale:* Captures complex, non-linear relationships missediques:** Trained a RandomForestClassifier on features comparing protected-original pairs (value differences, ratios, categorical agreement, initial similarity score). Used high by simple similarity, improving accuracy on ambiguous cases.

4.  **Transformation/Noise Analysis:**
    *   Analy-confidence pairs as positive examples and random pairs as negative examples. Predicted match probabilities for lower-confidence candidates identified previously.
    *   **Rationalezed high-confidence pairs to identify systematic transformations (scaling, shifting) and characterize residual noise.
    *   Calcul:** ML models can capture non-linear relationships and weigh features intelligently, improving discrimination between close candidates.

4.  **Transformation/Noise Analysis:**
    *   **Goal:** Understand *how* the data was likely altered to gain insight into the protection mechanism.
    *   **Techniques:** Analyzed differences (`protected - original`) and ratios (`protected / original`) for high-confidence matchesated differences (Protected - Original) and fitted statistical distributions (`norm`, `laplace` from `scipy.stats`) to the noise.
    *   *Rationale:* Understand *how* the data was altered to gain insight into the protection mechanism.

5.  **Distance-Based Refinement:**
    *   Leveraged the transformation findings (primarily minor scaling) to calculate "expected" original values based on protected values.
    *   Recalculated matches based on the minimum squared Euclidean distance between records and their expected original counterparts across relevant columns.
    *   *Rationale. Identified dominant transformation types (scaling, shifting, noise). Fitted Gaussian and Laplace distributions to the observed noise for key numerical columns.
    *   **Rationale:** Reverse-engineering the protection provides specific insights for targeted refinement and vulnerability assessment.

5.  **Distance-Based Refinement:**
    *   **Goal:** Leverage the specific transformation knowledge gained in the previous step to further refine matches.
    *   **Techniques:** Calculated "expected" original values by reversing the identified scaling/shifting. Found:* Uses specific knowledge of the data alteration to provide an alternative, targeted refinement method.

6.  **Vulnerability Quantification (Attempted):**
    *   Attempted to estimate the effective privacy budget (ε) based on fitted noise distributions (Laplace) and *explicitly stated assumptions* about data sensitivity (L1 sensitivity).
    *   *Rationale:* the original record with the minimum squared Euclidean distance to these expected values across multiple columns.
    *   **Rationale:** Directly applies insights about the specific data alterations, providing a complementary refinement logic to the ML approach.

6.  **Vulnerability Quantification (Epsilon Estimation):**
    *   **Goal:** Quantify the achieved privacy level using standard metrics.
    *   **Techniques:** Provides a standard metric to quantify the level of privacy achieved (or lack thereof), offering deeper analytical insight.

## 3. Key Results & Findings

Our analysis yielded the following significant results:

*   **Extremely High Linkage Rate:** The multi-stage approach achieved near-perfect re-identification of records.
    *   Initial High/Medium Confidence Rate: **~99% (98.91%)**.
    *   ML Refinement potentially corrected **~558-575** matches. *( Used the scale parameter from the fitted noise distributions (primarily Laplace) and *assumed* L1 data sensitivities (maximum possible difference between individuals) to estimate the effective Differential Privacy parameter epsilon (ε) for key columns.
    *   **Rationale:**Use your specific number)*
    *   Distance Refinement potentially corrected **~800** matches. *(Use your specific number)*
*   **Protection Method Identified:** The primary alterations detected were:
    *   Minor scaling (factor ≈ 1.00 Epsilon provides a standard measure of privacy loss; lower values imply stronger (intended) privacy. Estimated epsilon reveals the *actual* vulnerability.

## Results Summary

*   **Matching Accuracy:** Near-perfect re-identification achieved.
    *   Initial High/Medium Confidence Rate: **98.91%**.
    *   ML Refinement corrected **~550-580** potential mismatches.
    *   Distance Refinement corrected **~800** potential mismatches.
*   ) across most numerical columns.
    *   Addition of low-magnitude, zero-centered, Gaussian-like noise.
    *   `Loan_Repayment` exhibited more complex noise patterns.
*   **Minimal Obfuscation:** The noise analysis (visualized and quantified in the report) clearly showed that the magnitude of the random noise added was very small for**Protection Method Revealed:** Data primarily altered by minor scaling (factor ≈ 1.0) plus low-magnitude, zero-centered Gaussian-like noise. `Loan_Repayment` showed more complex alterations.
*   **Privacy Vulnerability:** The minimal most key financial variables.
*   **Privacy Vulnerability:** The minimal nature of the perturbations indicates **insufficient anonymization**. The data remains highly vulnerable to re-identification attacks, especially when auxiliary information (like the original dataset) is available. Formal Epsilon estimation was challenging due to noise patterns, but the low noise itself points to weak privacy guarantees.

## 4. Tech noise magnitude indicates insufficient obfuscation. While noise patterns didn't perfectly fit standard DP models for precise epsilon calculation, the analysis Stack & Libraries

*   **Language:** Python 3.x
*   **Core Libraries:**
    *   `pandas`: Data clearly demonstrates weak practical privacy.

*(Note: Exact refinement counts may vary slightly between runs due to ML randomness.)*

## Key manipulation.
    *   `numpy`: Numerical computation.
    *   `scikit-learn`: Scaling, Similarity Findings & Privacy Implications

1.  **High Linkage Feasibility:** The applied protection method was **insufficient** to prevent large, RandomForestClassifier.
    *   `scipy`: Statistical distribution fitting (norm, laplace).
    *   `matplotlib` & `seaborn`: Data visualization.
    *   `time`: Performance measurement.

## 5. Files-scale re-identification when original data context is available.
2.  **Superficial Protection:** The primary alterations in this Repository/Submission

*   `[your_code_script_name.py / .ipynb]`: The main involved minimal scaling and low-variance noise, failing to significantly obscure individual records.
3.  **Quantifiable Weak Python script/Jupyter Notebook containing the analysis code.
*   `protected_data_challenge.csv`: Theness:** The nature of the noise points towards weak privacy guarantees, far from what robust Differential Privacy mechanisms typically provide.

## input protected dataset.
*   `data-challenge-original.csv`: The input original dataset.
*   `matched_records_final.csv`: Output CSV containing the final linked pairs (Protected ID <-> Original ID) and Files in this Repository

*   `record_linkage_attack.ipynb` / `record_linkage_attack associated scores/confidence.
*   `full_analysis_report.txt`: Detailed text report summarizing results, transformations.py`: The main Jupyter Notebook or Python script containing the full analysis code.
*   `protected_data_challenge.csv`: The, noise analysis, and epsilon estimations.
*   `privacy_analysis_results_enhanced.png`: Composite image visualizing input protected dataset.
*   `data-challenge-original.csv`: The input original dataset.
*   ** key results (confidence, similarity, ML probs, transformations, noise distributions).
*   `hackathon_summary_results.pngOutput Files (Generated by the script):**
    *   `matched_records_final.csv`: The final list`: Single image summarizing the key findings and conclusions.
*   `README.md`: This file.

## 6. How linking protected record Identifiers to their matched original record Identifiers, including confidence scores and refinement details.
    *   `full_ to Run

1.  Ensure all required Python libraries (pandas, numpy, scikit-learn, scipy, matplotlib, seaborn) are installed.
    ```bash
    pip install pandas numpy scikit-learn scipy matplotlib seaborn
    ```
2.analysis_report.txt`: A detailed text report summarizing matching stats, transformation analysis per column, noise characteristics, and epsilon estimation  Place the input CSV files (`protected_data_challenge.csv`, `data-challenge-original.csv`) in the same directory results (including assumptions).
    *   `privacy_analysis_results_enhanced.png`: A composite image visualizing key as the script/notebook.
3.  Execute the Python script or run the cells sequentially in the Jupyter Notebook ` results (confidence distribution, similarity scores, ML probabilities, transformation types, noise histograms).
    *   `hackathon[your_code_script_name.py / .ipynb]`.
4.  The script will print progress_summary_results.png`: A single image summarizing the key takeaways and results graphically.
*   `README.md`: This file updates and generate the output files (`.csv`, `.txt`, `.png`) in the same directory.

## 7. Conclusion.

## Tech Stack

*   **Language:** Python 3.x
*   **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`

## How to Run

1.  Ensure

This project successfully demonstrated a high degree of vulnerability in the provided "protected" dataset. Our multi-stage attack, incorporating you have Python 3 installed along with the libraries listed in the Tech Stack (`pip install pandas numpy scikit-learn scipy advanced techniques like ML-based refinement and quantitative noise analysis, achieved near-complete re-identification. The findings indicate that the privacy matplotlib seaborn`).
2.  Place the input CSV files (`protected_data_challenge.csv`, `data-challenge--preserving techniques employed were insufficient, primarily involving only minor scaling and low-magnitude noise, failing to provide robust anonymity againstoriginal.csv`) in the same directory as the script/notebook.
3.  Run the `record_linkage_attack.ipynb` a targeted linkage attack.






Language: Python
Core Data Science Libraries: pandas , numpy , scikit-learn ,scipy .
Visualization Libraries: matplotlib , seaborn