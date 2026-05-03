# Lorenz Attractor MLP Project: Defense Questions

This document contains a curated set of questions for the in-person defense of the "Multilayer Perceptrons for Lorenz Attractor Prediction" project. Questions range from basic conceptual checks to deep technical probes.

---

## 1. Dynamical Systems & The Lorenz Attractor
1.  **Transient Behavior:** Why was it necessary to discard the first 1000 steps of the simulation? What would have happened to the model's training if you had included them?
2.  **Sensitivity to Initial Conditions:** If you changed the initial condition $(x, y, z)$ from $(0.1, 0.1, 0.1)$ to $(0.10001, 0.1, 0.1)$, how would the trajectory change over time, and how would this affect the MLP's ability to predict many steps into the future?
3.  **The Time Step ($\Delta t$):** Your code uses $\Delta t = 0.01$ with the Euler Method. What are the risks of using a significantly larger $\Delta t$, and how would that manifest in your dataset?

## 2. Model A: Manual Backpropagation (The "Scratch" Model)
1.  **The Chain Rule:** Walk us through the mathematical derivation of the gradient for the weight $W_1$ (input to hidden). Which partial derivatives are involved?
2.  **Sigmoid Saturation:** You used a Sigmoid activation. If your input data wasn't scaled (e.g., $x$ values around 20-30), what happens to the derivative of the Sigmoid, and how does that affect the weight updates?
3.  **Weight Initialization:** You implemented Xavier/Glorot initialization. Why is it dangerous to initialize all weights to zero in a neural network?
4.  **Bias Terms:** Why do we need the bias vectors $b_1$ and $b_2$? What geometric operation do they allow the neurons to perform?

## 3. Data Preprocessing & Scaling
1.  **Mixed Scaling Strategy:** You used `StandardScaler` for the inputs ($X$) but `MinMaxScaler` for the targets ($y$). What is the specific technical reason for forcing the targets into a $[0, 1]$ range?
2.  **Temporal Leakage:** When splitting the data into training and test sets, you set `shuffle=False`. Explain why shuffling would be a catastrophic mistake for this specific time-series problem.
3.  **Inverse Transformation:** Why is it necessary to use `.inverse_transform()` before calculating the final MSE or plotting the results?

## 4. Model B & Comparative Analysis
1.  **Architectural Capacity:** Model B has 16 hidden neurons, while Model A has only 3. Beyond just having "more neurons," how does this increased capacity allow Model B to better capture the "butterfly" transitions of the attractor?
2.  **The Adam Optimizer:** Model B uses the Adam optimizer. How does Adam differ from the standard Gradient Descent used in Model A regarding the "learning rate"?
3.  **Phase Shift:** In your plots for Model A, a "phase shift" (a delay in prediction) is visible. Why does a shallow/under-trained model often default to "predicting the current value as the next value"?

## 5. Validation & Overfitting
1.  **K-Fold on Time Series:** You implemented K-Fold Cross-Validation. Is a standard K-Fold (random folds) appropriate for time series, or did you use a specific temporal cross-validation strategy?
2.  **Generalization:** If your model has a very low MSE on the training set but a very high MSE on the test set, what is happening, and how would you fix it using "Regularization"?

## 6. Critical Thinking (Ruthless Tier)
1.  **The "Next Step" Fallacy:** Your model predicts $t+1$ given $t$. If we want to predict the state 100 steps into the future using only the current state, how would you modify your testing loop? What would happen to the error as you iterate?
2.  **Non-Compliance:** The project instructions required a team of 4-5 members. This team has 2. How did this impact your ability to explore more complex architectures (like LSTMs or GRUs) which are typically better suited for chaotic time series?
