# Heilmeier Catechism Draft
## ECE 410/510 Spring 2026 | Mohammad Alshaiji

---

### Q1: What are you trying to do?

Neural networks learn by sending error signals backward through every layer in sequence. This process cannot be fully parallelized across layers because each layer must wait for the layer above it to finish before it can update. I want to build a digital hardware accelerator that speeds up training using a different method called Direct Feedback Alignment (DFA). In DFA, the error signal is sent directly to all layers at once using fixed random matrices, removing the sequential layer dependency. The computationally dominant operation in DFA is matrix-vector multiplication (MVM). The goal is to design and synthesize a custom chiplet that accelerates this MVM kernel and demonstrate measurable speedup over a software baseline.

---

### Q2: How is it done today, and what are the limits of current practice?

Neural network training today uses backpropagation, which calculates how much each layer contributed to the error by working backward through the network one layer at a time. This sequential structure means layer N cannot begin updating until layer N+1 is done, and latency scales linearly with network depth. DFA removes this layer dependency by using fixed random matrices to project the output error directly to each layer, allowing weight updates across layers to be computed in parallel. Existing hardware implementations of DFA have focused on photonic and optoelectronic platforms. Existing work has not characterized how DFA's two distinct matrix operations, updatable forward weights and fixed feedback matrices, should be treated differently in a synthesized hardware architecture.

---

### Q3: What is new in your approach and why do you think it will be successful?

The new contribution is a synthesized MVM accelerator specifically designed for DFA training, implemented in SystemVerilog and synthesized through OpenLane 2 targeting the Sky130 process. The accelerator separates the storage and compute paths for the two types of matrix operations in DFA: the forward weight matrices, which are updated every training step, and the feedback matrices, which are fixed after initialization and never written again. These two paths have different memory access patterns and different write requirements, and a single unified architecture that treats them identically does not reflect those differences. The approach will be successful because the MVM kernel is well-defined, the arithmetic intensity can be computed directly from the algorithm, the software baseline can be implemented in pure NumPy without library dependencies, and the design can be synthesized and benchmarked through a standard RTL-to-GDSII flow. Each of these steps is independently verifiable, which keeps the project scope controlled and the results defensible.
