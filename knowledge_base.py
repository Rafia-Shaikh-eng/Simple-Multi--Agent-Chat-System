
# Minimal curated KB to support the sample scenarios
KB = {
    "neural networks": [
        "Feedforward neural networks (FNN) are the simplest architecture.",
        "Convolutional neural networks (CNN) specialize in grid-like data such as images; efficient feature sharing via kernels.",
        "Recurrent neural networks (RNN) model sequences with feedback connections.",
        "Transformers rely on self-attention, enabling parallelism and state-of-the-art results in NLP and vision.",
        "Graph neural networks (GNN) operate on graph-structured data."
    ],
    "transformers": [
        "Transformer architectures use self-attention to capture long-range dependencies efficiently.",
        "Multi-head attention allows the model to focus on different representation subspaces.",
        "Transformers are highly parallelizable, improving training throughput on GPUs/TPUs.",
        "Computational cost rises quadratically with sequence length in vanilla self-attention.",
        "Variants like Longformer, Performer, and Linformer reduce attention complexity with approximations."
    ],
    "reinforcement learning": [
        "Recent RL papers explore model-based RL for sample efficiency.",
        "Common challenges include exploration-exploitation trade-off, reward sparsity, and stability.",
        "Policy gradient methods optimize expected returns but can have high variance.",
        "Value-based methods like DQN approximate Q-values and are data-efficient in discrete spaces.",
        "Benchmarking differences and environment stochasticity hinder reproducibility."
    ],
    "optimizers": [
        "Gradient Descent iteratively updates parameters along negative gradients; simple and widely used.",
        "Adam combines momentum and adaptive learning rates; often converges faster and is robust.",
        "RMSProp adapts learning rates based on a moving average of squared gradients.",
        "Adagrad adapts learning rates per-parameter; can diminish over time.",
        "Second-order methods like L-BFGS can converge quickly on small to medium problems but are memory-intensive."
    ],
    "tradeoffs": [
        "Transformers scale well with data and compute but can be memory-hungry for long sequences.",
        "Efficient attention variants trade exactness for scalability, introducing approximation error.",
        "CNNs are efficient for local patterns; RNNs struggle with long-range dependencies compared to Transformers."
    ]
}
