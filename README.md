# PIN_EMNLP2020
The code and data for EMNLP 2020 paper: [Parallel Interactive Networks for Multi-Domain Dialogue State Generation](https://arxiv.org/pdf/2009.07616.pdf)

## Description
In this project, we focus on the multi-domain dialogue state tracking (MDST) problem. In the existing MDST models, the dependencies between system and user utterances in the same turn and across different turns are not fully considered (the example in Figure 1 shows the two kinds of dependencies). In this study, we argue that the incorporation of these dependencies is crucial for the design of MDST and propose Parallel Interactive Networks (PIN) to model these dependencies. Specifically, we integrate an interactive encoder to jointly model the in-turn dependencies and cross-turn dependencies. The slot-level context is introduced to extract more expressive features for different slots. And a distributed copy mechanism is utilized to selectively copy words from historical system utterances or historical user utterances. Empirical studies demonstrated the superiority of the proposed PIN model.

<p align="center">
  <img src="./fig/depend.png">
</p>
