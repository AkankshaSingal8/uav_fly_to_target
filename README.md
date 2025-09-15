# UAV Fly to Target 

This repository contains the dataset, code for model architecture and training and gazebo environment for experimentation. Model architecture, training hyper-parameters and NCP parameters.

### Creating Conda Environment
Clone this github repository
conda env create -f environment.yml
conda activate causality


### Table: NCP Model Architecture Summary

| **Layer (Type)**                        | **Activation** | **Param #**  |
|------------------------------------------|--------------|--------------|
| InputLayer                               | -            | 0            |
| Rescaling                                | -            | 0            |
| TimeDistributed (Normalization)         | -            | 7            |
| TimeDistributed (Conv2D)                 | ReLU         | 1,824        |
| TimeDistributed (Conv2D)                 | ReLU         | 21,636       |
| TimeDistributed (Conv2D)                 | ReLU         | 43,248       |
| TimeDistributed (Conv2D)                 | ReLU         | 27,712       |
| TimeDistributed (Conv2D)                 | ReLU         | 9,232        |
| TimeDistributed (Flatten)                | -            | 0            |
| TimeDistributed (Dense, 128 units)       | Linear       | 159,872      |
| TimeDistributed (Dropout)                | -            | 0            |
| RNN (NCP with LTC Cell)                  | -            | 22,398       |
| **Total Parameters**                     | -            | **285,929**  |
| **Trainable Parameters**                 | -            | **285,922**  |
| **Non-Trainable Parameters**             | -            | **7**        |

---

### Table: Backbone CNN Architecture

| **Layer Type**  | **Input Dim.**  | **Filters** | **Kernel Size** | **Stride** |
|----------------|----------------|-------------|-----------------|------------|
| 2D Conv.      | 144×256×3       | 24          | 5×5             | 2          |
| 2D Conv.      | 70×126×24       | 36          | 5×5             | 2          |
| 2D Conv.      | 33×61×36        | 48          | 5×5             | 2          |
| 2D Conv.      | 15×29×48        | 64          | 3×3             | 1          |
| 2D Conv.      | 13×27×64        | 16          | 3×3             | 2          |
| Flatten       | N/A             | N/A         | N/A             | N/A        |
| Fully Conn.   | 1248            | N/A         | N/A             | N/A        |


### Table: Neural Circuit Policies (NCP) Fixed Parameters

| **Parameter Name**               | **Value** |
|----------------------------------|-----------|
| Inter neurons                   | 18        |
| Command neurons                 | 12        |
| Motor neurons                   | 4         |
| Sensory fanout                  | 6         |
| Recurrent command synapses      | 4         |
| Motor fanin                     | 6         |


### Table: Training Hyper-parameters

| **Parameter Name** | **Value** |
|-------------------|-----------|
| Train epochs     | 100       |
| Optimizer       | ADAM      |
| Data shift      | 1         |
| Sequence length | 64        |
| Batch size      | 64        |
| LR             | 0.001     |
| LR Decay Rate  | 0.85      |

![image](https://github.com/user-attachments/assets/d4cf0343-529e-4ff4-bcb2-7077a235e38a)

Some of the code is referenced from: https://github.com/makramchahine/drone_causality/tree/main
