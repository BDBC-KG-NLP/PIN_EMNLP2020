# PIN_EMNLP2020
The code and data for EMNLP 2020 paper: [Parallel Interactive Networks for Multi-Domain Dialogue State Generation](https://arxiv.org/pdf/2009.07616.pdf)

## Description
In this project, we focus on the multi-domain dialogue state tracking (MDST) problem. In the existing MDST models, the dependencies between system and user utterances in the same turn and across different turns are not fully considered. In this study, we argue that the incorporation of these dependencies is crucial for the design of a MDST model. These dependencies exist because of the interactive nature of the dialogues. The interaction of the user and the system is often organized by a question-answering style. It is common in dialogue state tracking that a domain or slot being specified by one of the user or system, then the value being answered by the other. For example, in the dialogue in Figure 1, the user specifies a Restaurant domain, and the system answers a restaurant name Curry Garden. As is shown in Figure 1, both in-turn dependencies and cross-turn dependencies contribute to discovering slotv-alue pairs. Based on the the interactive nature of the dialogues, we build an Interactive Encoder which completely accords with the dependencies expressed in Figure 1 to jointly model the in-turn dependencies and cross-turn dependencies. As is shown in Figure 2, the Interactive Encoder consists of two parallel recurrent networks which interleave their hidden states.

<p align="center">
  <img src="./fig/dep.png" width="400"/> <img src="./fig/interact.png" width="430"/>
</p>

We also consider the slot overlapping problem in MDST. Unlike single-domain DST, slot overlapping is common in MDST and these overlapping slots share the similar values. For example, both Restaurant and Hotel domain have a slot price range which shares the same values. Under this condition, a generator without considering slotspecific features may mistakenly extract the value of one slot as the value of some other slot. To overcoming the slot overlapping problem, we introduce a slot-level context in the state generator. In addition, we also propose a distributed copy mechanism to selectively copy words from historical system utterances or historical user utterances. The overall structure of the proposed model is shown in Figure 3.

<p align="center">
  <img src="./fig/overall.png" width="700" />
</p>

Our implementation of the PIN model is based on the baseline model TRADE (Wu et al., 2019). We evaluate the models on the MultiWOZ2.0 dataset (Budzianowski et al., 2018) and MultiWOZ2.1 dataset (Eric et al., 2020). The evaluation results are shown in Table 1 and Table 2. The PIN model became a new state-of-the-art model on the MultiWOZ2.0 dataset. Although PIN performs worse than DST-Picklist on the MultiWOZ2.1 dataset, it improves upon the baseline model TRADE by a significant margin.

<p align="center">
  <img src="./fig/mwoz20.png" width="380"/> <img src="./fig/mwoz21.png" width="440"/>
</p>

## References
[Budzianowski et al., 2018] Pawel Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Inigo Casanueva, Stefan Ultes, Osman Ramadan and Milica Gasic. MultiWOZ-A Large-Scale Multi Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling. In EMNLP 2018.

[Wu et al., 2019] Chien-Sheng Wu, Andrea Madotto, Ehsan Hosseini-Asl, Caiming Xiong, Richard Socher and Pascale Fung. Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems. In ACL 2019.

[Eric et al., 2020] Mihail Eric, Rahul Goel, Shachi Paul, Abhishek Sethi, Sanchit Agarwal, Shuyang Gao and Dilek Hakkani-Tur. MultiWOZ 2.1: A Consolidated Multi-Domain Dialogue Dataset with State Corrections and State Tracking Baselines. In LREC 2020.
