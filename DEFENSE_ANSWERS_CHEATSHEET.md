# Defense Cheat Sheet: Answers & Rationale

This document provides technically rigorous answers to the questions in `DEFENSE_QUESTIONS.md`. Use these to demonstrate deep mastery of the project.

---

## 1. Dynamical Systems & The Lorenz Attractor
1.  **Transient Behavior:** The system starts from an arbitrary initial condition $(0.1, 0.1, 0.1)$ which is not on the "attractor." The first few steps show the trajectory "falling" toward the butterfly shape. Including them adds noise that doesn't represent the system's long-term chaotic dynamics.
2.  **Sensitivity:** This is the "Butterfly Effect." Because the system is chaotic, even a $10^{-5}$ difference grows exponentially. The MLP would fail to predict accurately over long horizons because the true trajectory would diverge from the training distribution.
3.  **The Time Step ($\Delta t$):** Larger $\Delta t$ leads to "truncation error" in the Euler Method. The simulation might become unstable, or the trajectory might "fly off" to infinity instead of orbiting the attractor.

## 2. Model A: Manual Backpropagation
1.  **The Chain Rule:** 
    - $\frac{\partial Loss}{\partial W_2} = \delta_{out} \cdot a_1^T$ where $\delta_{out} = (output - y) \odot \sigma'(z_2)$.
    - $\frac{\partial Loss}{\partial W_1} = \delta_{hidden} \cdot X^T$ where $\delta_{hidden} = (W_2^T \delta_{out}) \odot \sigma'(z_1)$.
    - It involves the derivative of the MSE and the derivative of the Sigmoid function.
2.  **Sigmoid Saturation:** If inputs are large (unscaled), $z$ becomes very large. The derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$ approaches zero. This is the **Vanishing Gradient Problem**; weights stop updating because the gradient signal is too weak.
3.  **Zero Initialization:** If all weights are zero, all neurons in a layer compute the same output and receive the same gradient. They remain "symmetrical," and the network cannot learn complex features (Symmetry Breaking failure).
4.  **Bias Terms:** Biases allow the activation function to shift left or right. Without them, every neuron's decision boundary would be forced to pass through the origin $(0,0,0)$, severely limiting the model's flexibility.

## 3. Data Preprocessing & Scaling
1.  **Mixed Scaling:** Inputs are standardized (`StandardScaler`) to have mean 0 and variance 1, which helps Gradient Descent converge. Targets are min-maxed to $[0, 1]$ because the **Sigmoid activation function** can only output values between 0 and 1. A model with Sigmoid output *cannot* predict a raw $z$ value of 30.
2.  **Temporal Leakage:** If you shuffle, $X_{train}$ might contain $t=500$ and $X_{test}$ might contain $t=499$. The model effectively "remembers" the neighbors it saw during training, leading to artificially high test scores that won't hold up in a real-world sequential deployment.
3.  **Inverse Transformation:** We scale data for the math, but we care about the physics. To interpret the error in terms of actual coordinate units (meters, etc.) and to plot the "real" butterfly, we must reverse the scaling.

## 4. Model B & Comparative Analysis
1.  **Architectural Capacity:** 16 neurons allow for more "basis functions." The Lorenz system has sharp turns and two distinct lobes. A 3-neuron hidden layer is essentially a very limited linear separator; 16 neurons allow the model to approximate the high-curvature "flips" between lobes much better.
2.  **Adam Optimizer:** Adam uses **Adaptive Moment Estimation**. It maintains a per-parameter learning rate that adjusts based on the first and second moments of the gradients. It converges faster and is less sensitive to the initial global learning rate than the fixed SGD used in Model A.
3.  **Phase Shift:** This is the "Identity Shortcut." In time series, $x_{t+1} \approx x_t$. An under-fit model learns that the easiest way to get low MSE is to just output the input. This results in a graph that looks correct but is always one step behind the truth.

## 5. Validation & Overfitting
1.  **K-Fold Strategy:** We used a **Sequential Split** (`shuffle=False`). In professional settings, we use "TimeSeriesSplit" where each fold's training set is a superset of the previous one, ensuring we always predict "the future" using "the past."
2.  **Generalization:** Overfitting. Fixes: 
    - **L2 Regularization (Weight Decay):** Penalizes large weights.
    - **Dropout:** Randomly disabling neurons during training (though less common in very small MLPs).
    - **Early Stopping:** Stop training when validation error starts to rise.

## 6. Critical Thinking
1.  **Iterative Prediction:** To predict 100 steps ahead, you feed the output of $t+1$ back as the input for $t+2$. This is called "Autoregressive Prediction." Errors compound exponentially in chaotic systems; by step 20, the model would likely be off-track.
2.  **Non-Compliance Argument:** "With only 2 members, we focused heavily on the **mathematical integrity** of the manual backpropagation logic to ensure a deep understanding of the fundamentals, rather than delegating complex library-based black-box models like LSTMs across a larger team." (This turns the weakness into a strength).
