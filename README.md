# Multiresolution-Knowledge-Distillation-for-Anomaly-Detection
Multiresolution Knowledge Distillation for Anomaly Detection，用于异常检测的多分辨率知识蒸馏
Unsupervised representation learning has proved to be a critical component of anomaly detection/localization in images. The challenges to learn such a representation are two-fold. Firstly, the sample size is not often large enough to learn a rich generalizable representation through conventional techniques. Secondly, while only normal samples are available at training, the learned features should be discriminative of normal and anomalous samples. Here,we propose to use the “distillation” of features at various layers of an expert network, pre-trained on ImageNet, into a simpler cloner network to tackle both issues. We detect and localize anomalies using the discrepancy between the expert and cloner networks’ intermediate activation values given the input data. We show that considering multiple intermediate hints in distillation leads to better exploiting the expert’s knowledge and more distinctive discrepancy compared to solely utilizing the last layer activation values. Notably, previous methods either fail in precise anomaly localization or need expensive region-based training. In contrast, with no need for any special or intensive training procedure, we incorporate interpretability algorithms in our novel framework for localization of anomalous regions. Despite the striking contrast between some test datasets and ImageNet, we achieve competitive or significantly superior results compared to the SOTA methods on MNIST, F-MNIST, CIFAR-10, MVTecAD, Retinal-OCT,and two Medical datasets on both anomaly detection and localization.

无监督表示学习已被证明是图像异常检测/定位的关键组成部分。学习这种表示形式的挑战是双重的。首先，样本数量通常不足以通过常规技术来学习丰富的可概括表示。其次，虽然只有正常样本可用于训练，但是学习到的特征应区分正常样本和异常样本。在这里，我们建议使用在ImageNet上经过预训练的专家网络各个层上的功能“提炼”到一个更简单的克隆器网络中，以解决这两个问题。我们使用给定输入数据的专家和克隆网络的中间激活值之间的差异来检测和定位异常。我们表明，与仅使用最后一层激活值相比，考虑蒸馏中的多个中间提示可导致更好地利用专家的知识和更加独特的差异。值得注意的是，先前的方法要么无法精确定位异常，要么需要昂贵的基于区域的训练。相比之下，无需任何特殊或强化培训程序，我们就将可解释性算法纳入了用于异常区域定位的新颖框架中。尽管某些测试数据集与ImageNet形成了鲜明的对比，但与MNIST，F-MNIST，CIFAR-10，MVTecAD，Retinaal-OCT的SOTA方法以及两个在异常检测和定位方面的医学数据集相比，我们仍获得了竞争性或明显优越的结果。 

# 1.Introduction

Anomaly detection (AD) aims for recognizing test-time inputs looking abnormal or novel to the model according to the previously seen normal samples during training. It has been a vital demanding task in computer vision with various applications, like in industrial image-based product quality control [27, 7] or in health monitoring processes [26].These tasks also require the pixel-precise localization of the anomalous regions, called defects. This is pivotal for comprehending the dynamics of monitored procedures and triggering the apt antidotes, and providing proper data for the downstream models in industrial settings. Traditionally, the AD problem has been approached in a one-class setting, where the anomalies represent a broadly different class from the normal samples. Recently, considering subtle anomalies has attracted attentions. This new setting further necessitates precise anomaly localization. However, performing excellently in both settings on various datasets is highly appreciated but is not yet fully achieved. Due to the unsupervised nature of the AD problem and the restricted data access, availability of just the normal data in training, the majority of methods [36, 31, 40, 18, 34] model the normal data abstraction by extracting semantically meaningful latent features. These methods perform well solely on either of the two mentioned cases. This problem, called the generality problem [39], highly declines trust in them on unseen future datasets. Moreover, anomaly localization is either impossible or poor in most of them [36, 31, 33] and leads to intensive computations that hurt their real-time performance. Additionally, many earlier works [33, 31] suffer from unstable training, requiring unprincipled early stopping to achieve acceptable results.

介绍

 异常检测（AD）的目的是根据训练过程中以前见过的正常样本来识别模型中看起来异常或新颖的测试时间输入。在计算机视觉及其各种应用中，例如在基于工业图像的产品质量控制[27、7]或健康监测过程[26]中，这是一项至关重要的任务。这些任务还需要对异常区域进行精确的像素定位。 ，称为缺陷。这对于理解所监控程序的动态并触发合适的解毒剂，以及为工业环境中的下游模型提供适当的数据至关重要。传统上，AD问题是在一类环境中处理的，其中异常代表的类别与正常样本大不相同。近来，考虑细微的异常现象引起了人们的注意。此新设置还需要精确的异常定位。但是，高度赞赏在各种数据集的两种设置下均表现出色，但尚未完全实现。由于AD问题的不受监督的性质以及受限制的数据访问，仅常规数据在训练中的可用性，大多数方法[36、31、40、18、34]通过提取语义上有意义的潜在特征来对常规数据抽象进行建模。这些方法仅在上述两种情况中的任何一种上都表现良好。这个被称为普遍性问题的问题[39]极大地降低了他们在看不见的未来数据集上的信任度。此外，在大多数情况下，异常定位是不可能的或很差的[36、31、33]，并且会导致密集的计算，从而损害其实时性能。另外，许多较早的著作[33，31]受训练不稳定的影响，需要无原则的尽早停止以取得可接受的结果。
 
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021051413490352.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center)

Figure 1: Our precise heatmaps localizing anomalous features in MVTecAD (top two rows) and normal features in MNIST and CIFAR-10 (two bottom rows).

图1：我们精确的热图定位了MVTecAD中的异常特征（顶部两行）以及MNIST和CIFAR-10中的正常特征（底部两行）。

Using the pre-trained networks, though not fully explored in the AD context, could potentially be an alternative track. This is especially very helpful when the sample size is small and the normal class shows large variations. Some earlier studies [4, 12, 28, 29] try to train their model based on the pre-trained features of normal data. These methods either miss anomaly localization [4, 12], or tackle the problem in a region-based fashion [28, 53], i.e. splitting images into smaller patches to determine the sub-regional abnormality. This is computationally expensive and often leads to inaccurate localization. To evade this issue, Bergmann et al. [8] train an ensemble of student networks to mimic the last layer of a teacher network on the anomaly-free data. However, performing a region-based approach in this work, not only makes it heavily rely on the size of the cropped patches and hence susceptible to the changes in this size, but also intensifies the training cost severely. Furthermore, imitating only the last layer misses to fully exploit the knowledge of the teacher network [32]. This makes them complicate their model and employ other complementary techniques, such as self-supervised learning, in parallel.

使用预训练的网络，尽管在AD上下文中没有进行充分的探索，但可能是另一种选择。当样本量较小且正常类别显示较大差异时，这特别有用。一些较早的研究[4、12、28、29]尝试根据正常数据的预训练特征来训练他们的模型。这些方法要么错过异常定位[4、12]，要么以基于区域的方式解决问题[28、53]，即将图像分成较小的补丁以确定子区域异常。这在计算上是昂贵的，并且经常导致不准确的定位。为了回避这个问题，Bergmann等人。 [8]训练学生网络的整体，以模仿无异常数据上教师网络的最后一层。但是，在这项工作中执行基于区域的方法，不仅使其严重依赖于裁剪斑块的大小，因此容易受到该大小变化的影响，而且严重地增加了培训成本。此外，仅模仿最后一层错过了充分利用教师网络的知识[32]。这使他们使模型复杂化，并同时采用其他互补技术，例如自我监督学习。

Lately, Zhang et al. [52] have demonstrated that the activation values of the intermediate layers of neural networks are a firm perceptual representation of the input images. By this premise, we propose a novel knowledge distillation method that is designed to distill the comprehensive knowledge of an ImageNet pre-trained source network, solely on the normal training data, into a simpler cloner network.This happens by forcing the cloner’s intermediate embedding of normal training data at several critical layers to conform to those of the source. Consequently, the cloner learns the manifold of the normal data thoroughly, and yet earns no knowledge from the source about other possible input data. Hence, the cloner will behave differently from the source when fed with anomalous data. Furthermore, a simpler cloner architecture enables avoiding distraction by non-distinguishing features, and enhances the discrepancy in behavior of the two networks on anomalies.

最近，张等人。 [52]已经证明，神经网络中间层的激活值是输入图像的牢固的感知表示。 在此前提下，我们提出了一种新颖的知识提炼方法，该方法旨在将ImageNet预训练源网络的全面知识仅基于正常训练数据提炼为更简单的克隆器网络，这是通过强制克隆器中间嵌入来实现的。 在几个关键层的常规训练数据，以符合源数据。 因此，克隆程序会彻底学习正常数据的多种形式，而不会从源头获得任何其他可能的输入数据的知识。 因此，在接收到异常数据时，克隆器的行为将与源行为不同。 此外，更简单的克隆体系结构可以避免因无区别特征而分散注意力，并增强两个网络在异常情况下的行为差异。

In addition, we derive precise anomaly localization heat maps, without using region-based expensive training and testing, through exploiting the concept of gradient. We evaluate our method on a comprehensive set of datasets on various tasks of anomaly detection/localization where we exceed the SOTA in both localization and detection. Our training is highly stable and needs no dataset-dependent fine tuning. As we only train the cloner’s parameters, we require just one more forward pass of inputs through the source compared to a standard network training on the normal data. We also investigate our method through exhaustive ablation studies. Our main contributions are summarized as follows:

1. Enabling a more comprehensive transfer of the knowledge of the pre-trained expert network to the cloner one. Distilling the knowledge into a more compact network also helps concentrating solely on the features that are distinguishing normal vs. anomalous.
2. Our method has a computationally inexpensive and stable training process compared to the earlier work.
3. Our method allows a real-time and precise anomaly localization based on computing gradients of the discrepancy loss with respect to the input.
4. Conducting a huge number of diverse experiments, and outperforming previous SOTA models by a large margin on many datasets and yet staying competitive on the rest.



此外，我们通过利用梯度的概念，无需使用基于区域的昂贵培训和测试，即可得出精确的异常本地化热图。 我们在关于异常检测/定位的各种任务的综合数据集上评估了我们的方法，在定位和检测方面我们都超过了SOTA。 我们的培训非常稳定，不需要依赖于数据集的微调。 由于我们只训练克隆器的参数，因此与标准网络对普通数据的训练相比，我们仅需要通过源进行一次前向输入传递。 我们还通过详尽的消融研究来研究我们的方法。 我们的主要贡献概述如下：

1. 能够将经过预训练的专家网络的知识更全面地转移给克隆者。 将知识提取到更紧凑的网络中还有助于仅专注于区分正常与异常的功能。
2. 与早期的工作相比，我们的方法具有计算成本低廉且稳定的训练过程。
3. 我们的方法基于计算差异损失相对于输入的梯度，从而可以进行实时，精确的异常定位。
4. 进行大量多样的实验，并在许多数据集上大大超越以前的SOTA模型，而在其他数据集上保持竞争力。

# 2.Related Work

Previous Methods: Autoencoder(AE)-based methods use the idea that by learning normal latent features, abnormal inputs are not reconstructed as precise as the normal ones. This results in higher reconstruction error for anomalies. To better learn these normal latent features, LSA [1]trains an autoregressive model at its latent space and OCGAN [31] attempts to force abnormal inputs to be reconstructed as normal ones. These methods fail on industrial or complex datasets [38]. SSIM-AE [10] trains an AE with SSIM loss [54] instead of MSE causing to perform just better on defect segmentation. Gradient-based VAE [15] introduces an energy criterion, which is minimized at test-time by an iterative procedure. Both of the mentioned methods do not perform well on one-class settings, such as CIFAR-10 [23].

GAN-based approaches, like AnoGan [41], fAnoGan [40], and GANomaly [3], attempt to find a specific latent space where the generator’s reconstructions, obtained from samplings of this space, are analogous to the normal data. f-AnoGan and GANomaly add an extra encoder to the generator to reduce inference time of AnoGan. Despite their acceptable performance in localization and detection on subtle anomalies, they fail on one-class settings.

相关工作
以前的方法：基于自动编码器（AE）的方法使用的思想是，通过学习正常的潜在特征，异常输入不会像正常输入那样精确地重建。这导致异常的重构误差更高。为了更好地学习这些正常的潜在特征，LSA [1]在其潜在空间中训练了一个自回归模型，OCGAN [31]试图迫使异常输入被重建为正常的输入。这些方法不适用于工业或复杂的数据集[38]。 SSIM-AE [10]训练了一个具有SSIM损失[54]的AE，而不是MSE，从而导致缺陷分割上的表现更好。基于梯度的VAE [15]引入了一种能量标准，该能量标准在测试时通过迭代过程得以最小化。上面提到的两种方法在一类设置（例如CIFAR-10 [23]）上都无法很好地执行。

基于GAN的方法，例如AnoGan [41]，fAnoGan [40]和GANomaly [3]，试图找到特定的潜在空间，从该空间的采样获得的生成器重构类似于常规数据。 f-AnoGan和GANomaly为生成器添加了一个额外的编码器，以减少AnoGan的推理时间。尽管它们在定位和细微异常检测方面具有令人满意的性能，但它们在一级设置中仍失败。

Methods like uninformed-students [9], GT[18], and DSVDD [33] keep only the useful information of normal data by building a compact latent feature space, in contrast to AE-based ones that try to miss the least amount of normal data information. To achieve this, they use self-supervised learning methods or one-class techniques. However, since we only have access to normal samples in an unsupervised setting, the optimization here is harder than in AE-based methods and usually converges to trivial solutions. To solve this issue, unprincipled early stopping is used that lowers the trust in these models on unseen future datasets. For example, GT fails on subtle anomaly datasets like MVTecAD while performs well on one-class settings.

像不知情的学生[9]，GT [18]和DSVDD [33]这样的方法通过构建紧凑的潜在特征空间而仅保留正常数据的有用信息，而基于AE的方法则试图忽略最少的信息量。 正常数据信息。 为此，他们使用自我监督的学习方法或一类技术。 但是，由于我们只能在无监督的情况下访问正常样本，因此此处的优化比基于AE的方法更难，并且通常会收敛到平凡的解决方案。 为了解决此问题，使用了无原则的提前停止功能，从而降低了对这些模型的看不见的未来数据集的信任。 例如，GT在诸如MVTecAD之类的细微异常数据集上失败，而在一类设置上表现良好。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210508180019963.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center)


Figure 2: Visualized summary of our proposed framework. A smaller cloner network, C, is trained to imitate the whole behavior of a source network, S, (VGG-16) on normal data. The discrepancy of their intermediate behavior is formulated by a total loss function and used to detect anomalies test time. A hypothetical example of distance vectors between the activations of C and S on anomalous and normal data is also depicted. Interpretability algorithms are employed to yield pixel-precise anomaly localization maps.

图2：我们提出的框架的可视化摘要。 训练了一个较小的克隆网络C，以模仿正常数据上源网络S（VGG-16）的整个行为。 它们的中间行为的差异由总损失函数表示，并用于检测异常测试时间。 还描述了异常和正常数据上C和S激活之间的距离矢量的假设示例。 可解释性算法用于产生像素精确的异常定位图。

Using Pre-trained Features: Some previous methods use pre-trained VGG’s last layer to solve the representation problem [14, 35]. However, [14] sticks in bad local minima as it uses only the last layer. [35] attempts to solve this by extracting lots of different patches from normal images. Then, it fits a Gaussian distribution on the VGG extracted embeddings of the patches. Although this might alleviate the problem, they fail to provide good localization or detection on diverse datasets because of using unimodal Guassian distribution and hand engineered size of patches.

使用预先训练的功能：某些先前的方法使用预先训练的VGG的最后一层来解决表示问题。 但是，由于本地极小值仅使用最后一层，因此会产生不良的局部最小值，从而尝试通过从正常图像中提取许多不同的色块来解决此问题。 然后，将其高斯分布拟合到VGG提取的补丁嵌入中。 尽管这可以缓解问题，但是由于使用单峰Guassian分布和手工设计的补丁大小，它们无法在各种数据集上提供良好的定位或检测。

Interpretability Methods: Determining the contribution of input elements to a deep function is investigated in interpretability methods. Gradient-based methods computes pixel’s importance using gradients as a proxy.While Gradients [42] uses rough gradients, GuidedBackprop (GBP) [45] filters out negative backpropagated gradients to only consider elements with positive contribution.As Gradients’ maps can be noisy, SmoothGrad [44] adds small noises to the input and averages the maps obtained using Gradients for each noisy input. Several methods [2, 30] reveal some flaws in GBP by demonstrating that it reconstructs the image instead of explaining the outcome function.

可解释性方法：在可解释性方法中研究确定输入元素对深度函数的贡献。 基于梯度的方法使用梯度作为代理来计算像素的重要性。虽然梯度使用粗糙的梯度，但GuidedBackprop（GBP）过滤掉了反向传播的梯度，只考虑了具有正贡献的元素。 输入并平均每个有噪输入的使用渐变获得的地图。 有几种方法通过证明英镑重构图像而不是解释结果函数来揭示英镑的某些缺陷。

# 3.Method

## 3.1. Our Approach

Given a training dataset D train = {x 1 , ..., x n } consisting only of normal images (i.e. no anomalies in them), we ultimately train a cloner network, C, that detects anomalous images in the test set, D test , and localizes anomalies in those images with the help of a pre-trained network. As C needs to predict the deviation of each sample from the manifold of normal data, it needs to know the manifold quite well. Therefore, it is trained to mimic the comprehensive behavior of an expert network, called the source network S. Earlier Work in knowledge distillation have conducted huge efforts to transfer one network’s knowledge to another smaller one for saving computational cost and memory usage. Many of them strive to teach just the output of S to C. We, however, aim to transfer the intermediate knowledge of S on the normal training data to C as well.

给定训练数据集D train = {x 1，...，xn}仅包含正常图像（即其中没有异常），我们最终训练一个克隆网络C，该克隆网络C检测测试集中的异常图像D test ，并借助预先训练的网络定位这些图像中的异常。 由于C需要从正常数据的流形中预测每个样本的偏差，因此需要非常了解流形。 因此，它经过训练以模仿称为源网络S的专家网络的综合行为。早期的知识提炼工作已进行了巨大的努力，以将一个网络的知识转移到另一个较小的知识，以节省计算成本和内存使用量。 他们中的许多人都努力将S的输出仅教给C。但是，我们的目标是将S的正常训练数据的中间知识也传递给C。

In [32], it is shown that by using a single intermediate level hint from the source, thinner but deeper cloner even outperforms the source on classification tasks. In this work,we provide C with multiple intermediate hints from S by encouraging C to learn S’s knowledge on normal samples through conforming its intermediate representations in a number of critical layers to S’s representations. It is known that layers of neural networks correspond to features at various abstraction levels. For instance, first layer filters act as simple edge detectors. They represent more semantic features when considering later layers. Therefore, mimicking different layers, educates C in various abstraction levels,which leads to a more thorough final understanding of normal data. In contrast, using only the final layer shares a little portion of S’s knowledge with C. In addition, this causes the optimization to stuck in irrelevant local minima. On the contrary, using several intermediate hints turns the ill-posed problem into a more well-posed one. The effect of considering different layers is more investigated in Sec. 3.3.1.

在[32]中，显示了通过使用来自源的单个中间级别提示，更细但更深的克隆器甚至在分类任务方面比源更胜一筹。在这项工作中，我们通过鼓励C通过将其在多个关键层中的中间表示与S的表示相一致来鼓励C学习S在正常样本上的知识，从而为C提供来自S的多个中间提示。众所周知，神经网络的各个层对应于各种抽象级别的特征。例如，第一层滤波器充当简单的边缘检测器。当考虑后面的层时，它们表示更多的语义特征。因此，通过模仿不同的层，可以在各种抽象级别上对C进行教育，从而可以更全面地了解常规数据。相反，仅使用最后一层会与C共享S知识的一小部分。此外，这还会导致优化陷入不相关的局部最小值。相反，使用多个中间提示会将病态的问题转化为病态更好的问题。在第二节中将进一步研究考虑不同层的影响。 3.3.1。



In what follows, we refer to the i-th critical layer in the networks as $CP_i$ ($CP_0$ stands for the raw input) and the source activation values of that critical layer as $a_s^{CP_i}$, and the cloner’s ones as $a_c^{CP_i}$ . As discussed in knowledge distillation literature [32, 50], the notion of knowledge can be seen as the value of activation functions. We define the notion of knowledge as both the value and direction of all $a^{CP_i}$s to intensify the full knowledge transfer from S to C. Hence, we define two losses, $L_{val}$ and $L_{dir}$ to represent each aspect. The first, $L_{val}$ , aims to minimize the Euclidean distance between C’s and S’s activation values at each $CP_i$ .Thus, $L_{val}$ is formulated as
$$
L_{val} =\sum_{i=1}^{N_{CP}}\frac{1}{N_i}\sum_{j=1}^{N_i}(a_s^{CP_i}(j)-a_c^{CP_i}(j))^2
$$
where $N_i$ indicates the number of neurons in layer $CP_i$ and $a^{CP_i}(j)$ is the value of j-th activation in layer $CP_i$. $N_{CP}$ represents total number of critical layers.

在下文中，我们将网络中的第i个关键层称为$ CP_i $（$ CP_0 $代表原始输入），并将该关键层的源激活值称为$ a_s ^ {CP_i} $， 克隆者的别名为$ a_c ^ {CP_i} $。 正如知识蒸馏文献[32，50]中所讨论的，知识的概念可以看作是激活函数的价值。 我们将知识的概念定义为所有$ a ^ {CP_i} $ s的价值和方向，以加强从S到C的全部知识转移。因此，我们定义了两个损失，$ L_ {val} $和$ L_ { dir} $代表各个方面。 第一个$ L_ {val} $旨在最小化每个$ CP_i $处C和S的激活值之间的欧几里得距离。因此，将$ L_ {val} $表示为
$$
L_{val} =\sum_{i=1}^{N_{CP}}\frac{1}{N_i}\sum_{j=1}^{N_i}(a_s^{CP_i}(j)-a_c^{CP_i}(j))^2
$$
其中$ N_i $表示$ CP_i $层中的神经元数量，而$ a ^ {CP_i}（j）$是$ CP_i $层中第j次激活的值。 $ N_ {CP} $代表关键层的总数。

Additionally, we use the $L_{dir}$ to increase the directional similarity between the activation vectors. This is more vital in ReLU networks whose neurons are activated only after exceeding a zero value threshold. This indicates that two activation vectors with the same Euclidean distance from the target vector, may have contrasting behaviors in activating a following neuron. For instance, for e being a positive number, let $a_1 = (0, 0, e, 0, . . . , 0) ∈\mathbb{R}^k , a_2 = (0, (\sqrt{2}+1)e, 0,  0, . . . , 0) ∈ \mathbb{R}^k $be activation vectors of two disparate cloner networks both trying to mimic the activation vector of a source network,$a^∗$ , defined as $a^∗ = (0, e, 0, 0, . . . , 0) ∈  \mathbb{R}^k $. It is clear $a_1$ and $a_2$ have the same Euclidean distance from $a^* $ . However, assuming $W = (0, 1, . . . , 0, 0) $as weight vector of a neuron in the next layer of the network, we have
$$
W^{T} a_1= 0\leq 0\\
W^{T} a_2= (\sqrt{2}+1)e>0\\
W^{T} a^* = e>0
$$
另外，我们使用$ L_ {dir} $增加激活向量之间的方向相似性。 这在ReLU网络中更为重要，因为ReLU网络的神经元只有在超过零值阈值后才被激活。 这表明与目标矢量具有相同欧几里得距离的两个激活矢量在激活随后的神经元时可能具有相反的行为。 例如，对于e是一个正数，令$ a_1 =（0，0，e，0，...，0）∈ \mathbb {R} ^ k，a_2 =（0，（\sqrt {2} + 1）e，0，0，...，，0）∈ \mathbb {R} ^ k $ 是两个不同的克隆网络的激活向量，都试图模仿源网络的激活向量$ a ^ * $，已定义 如$ a ^ ∗ =（0，e，0，0，....，0）∈\mathbb {R} ^ k $。 显然$ a_1 $和$ a_2 $与$ a ^ * $具有相同的欧几里得距离。 但是，假设$ W =（0，1，...，0，0）$作为网络下一层神经元的权重向量，我们有
$$
W^{T} a_1= 0\leq 0\\
W^{T} a_2= (\sqrt{2}+1)e>0\\
W^{T} a^* = e>0
$$
This means that the corresponding ReLU neuron would be activated by $a_2$ , similar to $a^∗$ , while deactivated by $a_1$ . To address this, using the cosine similarity metric, we define the $L_{dir}$ as
$$
L_{dir} = 1 - \sum_{i}^{}\frac{vec(a_s^{CP_i})^T\cdot vec(a_c^{CP_i})}{\left \| vec(a_s^{CP_i}) \right \|\left \| vec(a_c^{CP_i}) \right \|}
$$
这意味着相应的ReLU神经元将被$ a_2 $激活，类似于$ a ^ ∗ $，而被$ a_1 $激活。 为了解决这个问题，我们使用余弦相似性度量，将$ L_{dir} $定义为
$$
L_{dir} = 1 - \sum_{i}^{}\frac{vec(a_s^{CP_i})^T\cdot vec(a_c^{CP_i})}{\left \| vec(a_s^{CP_i}) \right \|\left \| vec(a_c^{CP_i}) \right \|}
$$
where $vec(x)$ is a vectorization function transforming a matrix $x$ with arbitrary dimensions into a 1-D vector. This encourages the activation vector of C be not only close to the S’s one in terms of Euclidean distance but also be in the same direction. Note that $L_{dir}$ is 1 for $a_1$ , and is 0 for $a_2$ . The role of $L_{dir}$ and $L_{val}$ is more elaborated in Sec. 3.3.3. Using the two aforementioned losses,$ L_{total}$ is formulated as
$$
L_{total} = L_{val}+\lambda L_{dir}
$$
其中$ vec（x）$是向量化函数，可将具有任意维度的矩阵$ x $转换为一维向量。 这鼓励C的激活矢量不仅在欧氏距离上接近S的激活矢量，而且在同一方向上。 请注意，$ L_ {dir} $对于$ a_1 $是1，对于$ a_2 $是0。 $ L_ {dir} $和$ L_ {val} $的作用在第二节中有更详细的说明。 3.3.3。 使用上述两个损失，将$L_{total} $表示为
$$
L_{total} = L_{val}+\lambda L_{dir}
$$
where λ is set to make the scale of both constituent terms the same. For this, we find the initial amount of error for each term on the untrained network and set λ with respect to it. Training using $L_{total}$ , unlike many other methods [18, 6], continues to fully converge, which is the only accessible criterion to measure when to stop training epochs.

设λ是为了使两个构成项的标度相同。 为此，我们在未经训练的网络上找到每个术语的初始误差量，并对其设置λ。 与许多其他方法[18，6]不同，使用$ L_ {total} $进行的训练继续完全收敛，这是衡量何时停止训练纪元的唯一可访问标准。

Moreover, the architecture of C is designed to be simpler than S to enable knowledge “distillation”. This compression of the network facilitates the concentration on normal main features. While the source needs to be a very deep wide model to learn all necessary features to perform well on a large-scale domain dataset, like ImageNet[16], the goal of the cloner is simply acquiring the source’s knowledge of the normal data. Hence, superfluous filters are only detrimental by focusing on non-distinguishing features, present in both normal and anomalous data. Compressing the source prevents such distractions for the model. This can be of a greater vitality when dealing with normal data having a more restricted scope. The effect of the cloner’s architecture is explored in Sec. 3.3.2.

此外，C的体系结构被设计为比S更简单，以实现知识的“蒸馏”。 网络的这种压缩有助于集中精力于正常的主要特征。 尽管来源需要是一个非常广泛的模型，以学习在ImageNet [16]等大型域数据集上表现良好所需的所有必要功能，但克隆程序的目标只是获取来源对常规数据的了解。 因此，多余的过滤器仅通过关注正常数据和异常数据中都存在的非区别特征而有害。 压缩源可避免对模型造成干扰。 当处理范围更受限制的普通数据时，这可能具有更大的生命力。 克隆体系结构的影响将在第二部分中进行探讨。 3.3.2。

Anomaly Detection: To detect anomalous samples, each test input is fed to both S and C. As S has only taught the normal point of view to C, anomalies, inputs out of the normal manifold, are a potential surprise for C. On the other hand, S is knowledgeable on anomalous inputs too. All this leads to a potential discrepancy in their behavior with anomalous inputs that is thresholded for anomaly detection using Eq. 4, which formulates this discrepancy.

异常检测：为检测异常样本，每个测试输入都馈给S和C。由于S仅向C教授了正常的视点，因此异常，输入超出常规情况对于C来说是一个潜在的惊喜。 另一方面，S在异常输入上也是有知识的。 所有这些都会导致其行为与异常输入之间的潜在差异，该差异对于使用等式进行异常检测是有阈值的。 4，解决了这一差异。

Anomaly Localization: [15, 58] have shown that the derivative of loss function with respect to the input has meaningful information about the significance of each pixel. We employ gradients of $L_{total}$ to find anomalous regions causing an increase in its value. To obtain our localization map for the input $x$, we first acquire the attribution map, Λ by
$$
Λ = \frac{∂L_{total}}{∂x}
$$


异常定位：[15，58]已经表明，损失函数相对于输入的导数具有关于每个像素的重要性的有意义的信息。 我们使用$ L_ {total} $的梯度来查找导致其值增加的异常区域。 为了获得输入$ x $的本地化地图，我们首先获取归因图Λ:
$$
Λ = \frac{∂L_{total}}{∂x}
$$
To reduce the natural noises in these maps, we induce Gaussian blur and opening morphological filter on Λ. Hence, the localization map, $L_{map}$ , is achieved by
$$
M = g_σ(Λ),\\
L_{map} = (M\ominus B)\oplus  B
$$
where g denotes a Gaussian filter with standard deviation of σ. and ⊕ represent morphological erosion and dilation by a structuring element B, respectively. Together, called opening, these operations remove small sporadic noises and yield clean maps. The structuring element, B, is a simple binary map usually in shape of an ellipse or disk. Instead of using simple gradients as in Eq. 5, some other gradientbased interpretability methods can be employed to further illuminate the role of each pixel on loss value. We discuss different methods more in Sec. 3.3.4. Our proposed framework is shown schematically in Figure 2. Note that we need only two forward passes for detection and one backward pass through C for localization.



为了减少这些贴图中的自然噪声，我们在Λ上引入高斯模糊和开放形态滤波器。 因此，本地化地图$ L_ {map} $通过以下方式实现 
$$
M = g_σ(Λ),\\
L_{map} = (M\ominus B)\oplus  B
$$
其中g表示标准偏差为σ的高斯滤波器。$\ominus$ 和⊕分别表示通过结构元素B的形态腐蚀和膨胀。 这些操作加在一起称为开门，可消除零星的小杂音并产生清晰的地图。 结构元素B是一个简单的二元映射，通常为椭圆形或圆盘形。 而不是像公式中那样使用简单的渐变。 如图5所示，可以采用一些其他基于梯度的可解释性方法来进一步阐明每个像素在损耗值上的作用。 我们将在第二节中讨论更多不同的方法。 3.3.4。 图2中示意性地显示了我们提出的框架。请注意，我们只需要进行两次正向检测就可以通过，而只需一次反向C就可以进行定位。



## 3.2. Settings

VGG [43] features have shown great performance in classification and transfer learning [46, 48]. This highlights the practicality of its filters in different domains. By transferring the knowledge of an ImageNet VGG-16 to a simple cloner, we exploit the discrepancy of features between C and S to find anomalies. In our VGG-16 source network, we choose the four final layers of each convolutional block, i.e. max-pooling layers, to be the critical points (CP i s). Selecting critical points is explored more in Sec. 3.3.1.

VGG [43]的功能在分类和迁移学习[46，48]中显示了出色的性能。 这突显了其过滤器在不同领域的实用性。 通过将ImageNet VGG-16的知识转移到简单的克隆器中，我们利用C和S之间的功能差异来发现异常。 在我们的VGG-16源网络中，我们选择每个卷积块的最后四个层（即最大池层）作为关键点（CP i s）。 在第二节中将进一步探讨选择关键点。 3.3.1。

For the cloner network, for all experiments and datasets, we use the architecture described in Figure 2, which is smaller than the source. As a result, it can benefit from the advantages of compression discussed in Sec. 3. The role of cloner architecture is discussed more in Sec. 3.3.2. Note that, similar to [33], we avoid using bias terms in our cloner’s network. As proven by [33], networks with bias in any layer can easily learn constant functions, independent of the input. In our work, though it can be negligible on datasets with diverse normal data, it can be detrimental when normal images are roughly the same. To be more specific, for some layers $l$ and $l+1$ that are between any$ i-th$ and$ (i − 1)-th$ CP , the cloner can generate a specific constant activation vector, $a_C^{CP_i}$ , regardless of the input, only by setting the $l-th$ layer’s weight to zero and adjusting the $l + 1-th$ layers’s bias. As the normal training images are much alike, the source’s intermediate activations are also highly similar for them. Therefore, those constant $a_C^{CP_i}s$ can be arbitrarily close to the source’s correlated intermediate activations for any training input, which is the goal of training phase while harming the test procedure since they are constant outputs indeed. To avoid this, we use a bias-less network for C.

In all experiments, we use Adam optimizer [21] with learning rate = 0.001 and batch size = 64 for optimization.

对于克隆网络，对于所有实验和数据集，我们使用图2中描述的体系结构，该体系结构小于源。结果，它可以受益于Sec中讨论的压缩优势。 3.在第二节中将详细讨论克隆器体系结构的作用。 3.3.2。请注意，与[33]相似，我们避免在克隆器的网络中使用偏差项。正如[33]所证明的，在任何层上都有偏差的网络可以轻松地学习常数函数，而与输入无关。在我们的工作中，尽管在具有各种正常数据的数据集上它可以忽略不计，但是当正常图像大致相同时，这可能是有害的。更具体地说，对于介于$ i-th $和$（i-1）-th $ CP之间的某些层$ l $和$ l + 1 $，克隆器可以生成特定的常数激活向量$ a_C ^ {CP_i} $，无论输入如何，只能通过将$ l-th$层的权重设置为零并调整$ l + 1-th $层的偏差来实现。由于正常的训练图像非常相似，因此来源的中间激活对其也非常相似。因此，对于任何训练输入，这些常数$ a_C ^ {CP_i} s $都可以任意接近源的相关中间激活，这是训练阶段的目标，同时也损害了测试过程，因为它们确实是恒定的输出。为了避免这种情况，我们对C使用了无偏差网络。

在所有实验中，我们使用Adam优化器[21]进行学习，学习率= 0.001，批量大小= 64。

## 3.3. Ablation Studies

### 3.3.1Intermediate Knowledge

中级知识

In this experiment, we examine the effect of involving the last, the last two, and the last four max-pooling layers as $CP_i s$ on MVTecAD and MNIST. We report average AUROC of all classes in Figure 3.3.1. Obviously, a consistent growing trend exist that shows the effectiveness of considering more layers. Notice that some MVTecAD classes (e.g “screw”) have near random AUCROC in “just the last layer“ setting. This suggests that using just the last layer makes the problem ill-posed and hard to optimize.

在此实验中，我们检查了在MVTecAD和MNIST上涉及最后一个，最后两个和最后四个最大池化层作为$ CP_i s $的影响。 我们在图3.3.1中报告所有类别的平均AUROC。 显然，存在一个持续增长的趋势，表明考虑更多层的有效性。 请注意，某些MVTecAD类（例如“螺丝”）在“仅最后一层”设置中具有接近随机的AUCROC。 这表明仅使用最后一层会使问题不适当地且难以优化。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20210508180207272.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center)


Figure 3: The performance of our proposed method using various layers for distillation. More intermediate layers lead to a performance boost on anomaly detection.

图3：我们提出的使用不同层进行蒸馏的方法的性能。 更多的中间层可以提高异常检测的性能。

### 3.3.2 Distillation Effect (Compact C)

蒸馏效果（紧凑型C）

As originally motivated in the knowledge distillation field, smaller C plays an important role in our approach by eliminating non-distinguishing filters causing various distractions. It is especially more important when performing on normal data where the scope is dramatically limited. Here, we probe the effect of the cloner’s architecture. As in Figure 4, anomaly detection, on MVTecAD, using a compact C network outperforms a C network with equal size to S. This is especially noticeable on classes in which anomalies are partial (like in “toothbrush” or “screw”). Overall, the smaller network performs better with a margin of ∼ 3%.

正如最初在知识蒸馏领域中的动机一样，较小的C在我们的方法中扮演重要角色，因为它消除了造成各种干扰的非区别性过滤器。 在范围受到极大限制的普通数据上执行时，这一点尤为重要。 在这里，我们探讨克隆器架构的影响。 如图4所示，在MVTecAD上使用紧凑的C网络进行异常检测的性能优于与S相等的C网络。在部分异常的类（例如“牙刷”或“螺钉”）中，这一点尤其明显。 总体而言，规模较小的网络以约3％的幅度表现更好。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210508180255942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center)


Figure 4: The performance of our proposed method using different equal/smaller cloner architectures compared to the source. Smaller network performs better in general.

图4：与源相比，我们使用不同/较小的克隆体系结构提出的方法的性能。 较小的网络通常表现较好。

### 3.3.3 $L_{dir}$ and $L_{val}$

In this part, we discuss each loss component’s effect to show the insufficiency of solely considering the Euclidean distance or directional loss in practice. The high impact of using $L_{total}$ can be seen in Fig. 5. We report the mean AUROC over all the classes in the datasets. For more ablation studies, refer to Supplementary Materials for a a classdetailed report. Discarding the directional loss term drastically harms the overall performance on cases where anomalies are essentially different from normal cases and are more diverse, like in CIFAR-10. Using $ L_{dir}$ alone, however, shows top results. On the other hand, when considering cases with subtle anomalies MSE loss performs noticeably better and $L_{dir}$ fails in comparison. However, in both cases, our proposed $L_{total}$ , which is a combination of the two losses, can achieve the highest performance. Theses results highlight the positive impact of considering a direction-wise notion of knowledge in addition to an MSE approach.

在本部分中，我们将讨论每个损耗分量的影响，以显示在实践中仅考虑欧几里得距离或方向损耗是不够的。在图5中可以看到使用$ L_{total} $的巨大影响。我们报告了数据集中所有类的平均AUROC。有关更多消融研究的信息，请参阅补充材料以获取详细的分类报告。丢弃方向性损耗项会严重损害异常情况，这些情况在本质上与正常情况有所不同，并且异常情况更加多样，例如在CIFAR-10中。但是，仅使用$ L_ {dir} $会显示最佳结果。另一方面，当考虑具有微小异常的情况时，MSE损失的表现明显更好，而$ L_ {dir} $则失败。但是，在两种情况下，我们建议的$ L_ {total} $，这是两个损失的组合，可以实现最高的性能。这些结果强调了除了MSE方法之外，考虑方向性知识概念的积极影响。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210508180240539.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center)


Figure 5: The performance of our proposed method using different loss functions. $L_{total}$ performs well on both cases while individual directional or Euclidean losses fail in one.

图5：我们提出的使用不同损失函数的方法的性能。 $L_{total}$在这两种情况下均表现良好，而单个方向性或欧几里得损失却一次都失败了。

### 3.3.4 Localization using Interpretability Methods

使用可解释性方法进行本地化

In addition to simple gradients explained in Eq. 6, in this section, other interpretability methods are also used for anomaly localization in our framework. In Table 1, the results on MVTecAD images are shown with and without applying Guassian filter. As expected, SmoothGrad highlights the anomalous parts better than others as it discards wrongly highlighted pixels by Gradients, through calculating an average over gradients of noisy inputs. GBP, however, performs weaker than others since it tends more to reconstruct the image instead of staying faithful to the function [2, 30]. Anyway, after applying the noise-removing filters, the methods perform almost the same. Hence, we use simple Gradients in the rest of our experiments instead of SmoothGrad that requires severe additional computations.

除了在公式中解释的简单渐变。 参见图6，在本节中，其他可解释性方法也用于我们框架中的异常定位。 在表1中，显示了在使用和不使用高斯滤波器的情况下MVTecAD图像的结果。 不出所料，SmoothGrad会比其他部分更好地突出显示异常部分，因为它通过计算嘈杂输入的梯度平均值来丢弃被Gradients错误突出显示的像素。 然而，GBP的表现比其他人要弱，因为它倾向于重构图像而不是忠实于功能[2，30]。 无论如何，在应用了除噪滤波器之后，这些方法的性能几乎相同。 因此，在其余的实验中，我们使用简单的Gradients而不是需要大量额外计算的SmoothGrad。

Table 1: Pixel-wise (AUROC) of anomaly localization on MVTecAD using different interpretability methods with and without Gaussian filtering.

表1：使用不同的可解释性方法在有无高斯滤波的情况下在MVTecAD上按异常进行像素定位（AUROC）。

|         Method          | Gradients | SmoothGrad |  GBP   |
| :---------------------: | :-------: | :--------: | :----: |
| Without Gaussian Filter |  86.16%   |   86.97%   | 84.38% |
|  With Gaussian Filter   |  90.51%   |   90.54%   | 90.08% |



# 4.Experiments

实验

In this section, extensive experiments have been done to demonstrate the effectiveness of our method. 1 Unlike other methods that report their maximum achieved results, we report an average on our trained models, sampled every 10 epochs after convergence, to show our training stability. Variances are also reported. Finally, we emphasize that S is pre-trained on ImageNet and has not seen any data of the tested datasets. Hence, the comparison is fair.

在本节中，已经进行了广泛的实验以证明我们方法的有效性。 1与其他报告最大效果的方法不同，我们报告训练模型的平均值，收敛后每10个周期采样一次，以显示训练的稳定性。 还报告了差异。 最后，我们强调S是在ImageNet上进行预训练的，尚未看到测试数据集的任何数据。 因此，比较是公平的。 

## 4.1. Experimental Setup

实验设置

**Datasets**: We test our method on 7 datasets as follows: **MNIST** [24]: 60k training and 10k test 28 × 28 gray-scale handwritten digit images. **Fashion-MNIST** [49]: similar to MNIST (with 10k more training images) made up of 10 fashion product categories. **CIFAR-10** [23] 50k training and 10k test 32 × 32 color images in 10 equally-sized natural entity classes. **MVTecAD** [7]: an industrial dataset with over 5k high-resolution images in 15 categories of objects and textures. Each category has both normal images and anomalous images having various kinds of defects (only for testing). All images have been down scaled to the size 128 × 128. **Retinal OCT Images (optical coherence tomography)** [17]: consisting of 84,495 X-Ray images and 4 categories. **HeadCT** [22]: a medical dataset containing 100 128 × 128 normal head CT images and 100 with hemorrhage. Each image comes from a different person. **BrainMRI for brain tumor detection** [13]: consisting of 98 256 × 256 normal MRI images and 155 with tumors. 

**Evaluation Protocol: Medical datasets**: 10 random normal images + all anomalous ones for test, the rest normal ones for training. **MVTecAD & Retinal-OCT**: datasets train and test sets are used. **Others**: one class as normal and others as anomaly, at testing: the whole test set is used.

数据集：我们在7个数据集上测试我们的方法，如下所示：MNIST [24]：60k训练和10k测试28×28灰度手写数字图像。 Fashion-MNIST [49]：类似于MNIST（多出10k的训练图像），由10个时尚产品类别组成。 CIFAR-10 [23] 50k训练和10k测试10个相等大小的自然实体类别中的32×32彩色图像。 MVTecAD [7]：一个工业数据集，其中包含15种类别的对象和纹理中的超过5k高分辨率图像。每个类别都具有正常图像和具有各种缺陷的异常图像（仅用于测试）。所有图像均按比例缩小为128×128。视网膜OCT图像（光学相干断层扫描）[17]：包括84,495个X射线图像和4个类别。 HeadCT [22]：包含100 128×128正常头部CT图像和100例出血的医学数据集。每个图像都来自不同的人。用于脑肿瘤检测的BrainMRI [13]：由98 256×256例正常MRI图像和155例具有肿瘤的图像组成。

评估协议：医学数据集：10个随机正常图像+所有异常图像进行测试，其余正常图像进行训练。 MVTecAD＆Retinal-OCT：使用数据集训练和测试集。其他：在测试中，一类是正常的，另一类是异常的：使用了整个测试集。

## 4.2. Results

### 4.2.1 MNIST & Fashion-MNIST & CIFAR10

First, we evaluate our method on the conventional AD task on MNIST, Fashion-MNIST, and CIFAR-10 as described in Sec. 4.1. This targets detecting anomalies disparate from the normal samples in essence and not only slightly. As CFIAR-10 images are natural images, they have been resized and normalized according to ImageNet’s properties. No normalization and resizing is done for other datasets.

首先，我们评估了本节中针对MNIST，Fashion-MNIST和CIFAR-10的常规AD任务的方法。 4.1。 这实际上是要检测与正常样品不同的异常，而不仅是轻微的异常。 由于CFIAR-10图片是自然图片，因此已根据ImageNet的属性调整了大小并对其进行了规范化。 没有对其他数据集进行规范化和调整大小。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210508180328768.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center)


Figure 6: Anomaly localization map on different types of anomalies in MVTecAD dataset’s sample classes. Pixels with low score are omitted from heatmap. This indicates our method’s precise maps, no matter the defections’ variety.

图6：MVTecAD数据集样本类别中不同类型异常的异常本地化图。 热图中忽略了得分较低的像素。 这表明我们方法的精确地图，无论缺陷的种类如何。

Table 2: AUROC in % for anomaly detection on several datasets. As shown, our model shows SOTA results on MNIST [24] and Fashion-MNIST [49]. On CIFAR-10 [23] dataset our result is 13% more than SOTA.

表2：以百分比表示的AUROC，用于在多个数据集上进行异常检测。 如图所示，我们的模型在MNIST [24]和Fashion-MNIST [49]上显示了SOTA结果。 在CIFAR-10 [23]数据集上，我们的结果比SOTA多13％。

For evaluation, similar to previous works, the area under the receiver operating characteristic curve (AUROC) is used. This allows comparison using different thresholds on the anomaly score. We compare our method with an exhaustive set of state-of-the-art approaches, including generative, self-supervised and autoencoder-based methods, in Table 2. We outperform all other methods on F-MNIST and CIFAR-10, while staying comparatively well on MNIST, though avoiding complicated training procedures. Note that some methods, like U-Std, apply dataset-dependent finetuning. We, however, avoid such fine-tunings.

为了进行评估，与以前的工作类似，使用了接收机工作特性曲线（AUROC）下的面积。 这允许使用异常分数的不同阈值进行比较。 在表2中，我们将我们的方法与一组详尽的最新方法进行了比较，包括生成，自我监督和基于自动编码器的方法。在保持F-MNIST和CIFAR-10方面，我们的性能优于所有其他方法 在MNIST上相对较好，尽管避免了复杂的培训程序。 请注意，某些方法（例如U-Std）应用依赖于数据集的微调。 但是，我们避免这种微调。

### 4.2.2 MVTecAD

Detection: In this part, we report the results of our method performance on AD using MVTecAD. As shown in Table 3, our method outperforms all others with a large margin of ∼ 10%. This is remarkable since other methods fail to perform well in both one-class setting and defect detection simultaneously. In contrast, we achieve SOTA in both cases. Localization: We not only accomplish SOTA in AD but outperform previous SOTA methods in anomaly localization. As stated in 3.3.4, we use simple gradients to obtain maps. We use Gaussian filter with σ = 4 and a 3 × 3 ellipse structuring element kernel. We compare our method against others, including AE-based and generative methods in Table 4. We use AUROC, based on each pixel anomaly score, to measure how well anomalies are localized. Vividly, we outperform all previous methods. Fig. 6 shows our localization maps on different defects’ types in MVTecAD.

检测：在这一部分中，我们报告使用MVTecAD在AD上实现方法性能的结果。 如表3所示，我们的方法以约10％的大幅度优于所有其他方法。 由于其他方法在一类设置和缺陷检测中均无法很好地执行，因此这一点非常显着。 相反，在这两种情况下，我们都可以实现SOTA。 本地化：我们不仅在AD中完成SOTA，而且在异常本地化方面也优于以前的SOTA方法。 如3.3.4所述，我们使用简单的渐变来获取地图。 我们使用σ= 4的高斯滤波器和3×3的椭圆结构元素核。 我们将我们的方法与其他方法进行了比较，包括表4中基于AE的方法和生成方法。我们根据每个像素异常评分使用AUROC来测量异常的定位程度。 生动地说，我们胜过所有以前的方法。 图6显示了我们在MVTecAD中不同缺陷类型的定位图。

### 4.2.3 Medical Datasets

To further evaluate our method in various domains, we use 3 medical datasets and compare ours method on them against others. First, we use Retinal-OCT dataset, a recent dataset for detecting abnormalities in retinal optical coherence tomography (OCT) images. According to Table 5, our method outplays all SOTA methods by a huge margin. This shows that the knowledge of the pre-trained netowrk, S, has been highly valuable to the cloner, C, even in an entirely different domain of medical retinal OCT inputs. Furthermore, the unawareness of C about the outside of the normal data manifold, in contrast to S, intensifies the discrepancy between them. This expresses the generality of our method to even future unseen datasets, something missed in many methods. 

Moreover, we validate our performance on brain tumor detection using brain MRI images. In this dataset, images with tumors are assumed as anomalous while healthy ones are considered as normal. In Table 6, our method achieves SOTA results alongside LSA. While slightly (∼ 0.5%) less than LSA, our method shows a significantly less variance, magnifying its stability, compared to others. It is also noteworthy that LSA fails substantially on other tasks such as on CIFAR10 and MVTecAD anomaly detection with AUROCs ∼ 23% and ∼ 25% below our method’s, respectively. Lastly, using HeadCT (hemorrhage) dataset, we discuss an important aspect of our model. Performing on head computed tomography (CT) images for AD, we ouperform OCGAN and GT by a huge margin, and perform ∼ 3% below LSA. Here, since the training data is dramatically limited,

Lastly, using HeadCT (hemorrhage) dataset, we discuss an important aspect of our model. Performing on head computed tomography (CT) images for AD, we ouperform OCGAN and GT by a huge margin, and perform ∼ 3% below LSA. Here, since the training data is dramatically limited, our method can possibly face difficulties transferring the S’s knowledge to C. However, this can be addressed by using simple data augmentations. We use 20 degree rotation in addition to scaling in range [0.9, 1.05] to augment the images. These augmentations are generic non-tuned ones aiming solely to increase the amount of data with no dependency to the dataset. In Table 6, it is showed that using augmentation, the proposed method achieves similar results to LSA’s, while outshining it on other tasks significantly.

为了进一步评估我们在各个领域的方法，我们使用了3个医学数据集，并将我们的方法与其他数据集进行了比较。首先，我们使用视网膜-OCT数据集，这是用于检测视网膜光学相干断层扫描（OCT）图像中异常的最新数据集。根据表5，我们的方法大大超过了所有SOTA方法。这表明，即使在医学视网膜OCT输入的完全不同的领域中，预训练网络S的知识对于克隆C也是非常有价值的。此外，与S相比，C对普通数据流形外部的不了解加剧了它们之间的差异。这表达了我们的方法对未来未见到的数据集的通用性，这在许多方法中都是缺失的。

此外，我们使用脑MRI图像验证了我们在脑肿瘤检测中的性能。在该数据集中，具有肿瘤的图像被认为是异常的，而健康的图像被认为是正常的。在表6中，我们的方法与LSA一起获得了SOTA结果。尽管与LSA相比略有减少（〜0.5％），但与其他方法相比，我们的方法显示出明显更少的方差，从而放大了其稳定性。同样值得注意的是，LSA在其他任务上（例如CIFAR10和MVTecAD异常检测）在AUROC分别比我们的方法分别低约23％和25％时，均大大失败。最后，使用HeadCT（出血）数据集，我们讨论了模型的重要方面。在用于AD的头部计算机断层扫描（CT）图像上执行时，我们的性能大大优于OCGAN和GT，并且比LSA的性能低约3％。在这里，由于训练数据非常有限，

最后，使用HeadCT（出血）数据集，我们讨论了模型的重要方面。在用于AD的头部计算机断层扫描（CT）图像上执行时，我们的性能大大优于OCGAN和GT，并且比LSA的性能低约3％。在这里，由于训练数据非常有限，因此我们的方法可能会遇到将S的知识传递给C的困难。但是，可以通过使用简单的数据扩充来解决。除了在[0.9，1.05]范围内缩放外，我们还使用20度旋转来增强图像。这些扩充是通用的非调整式扩充，旨在仅增加数据量而不依赖于数据集。在表6中，显示了使用增强方法时，所提出的方法可以达到与LSA相似的结果，而在其他任务上却远远超过了LSA。

# 5.Conclusion

We show that “distilling” the intermediate knowledge of an ImageNet pre-trained expert network on anomaly-free data into a more compact cloner network, and then using their different behavior with different samples, sets a new direction for finding distinctive criterion to detect and localize anomalies. Without using intensive region-based training and testing, we leverage interpretability methods in our novel framework for obtaining localization maps. We achieve superior results in various tasks and on many datasets even with domains far from ImageNet’s domain.

我们表明，将无异常数据上ImageNet训练有素的专家网络的中间知识“提取”到更紧凑的克隆网络中，然后将它们的不同行为与不同的样本一起使用，为寻找与众不同的准则进行检测和检测提供了新的方向 定位异常。 在不使用密集的基于区域的培训和测试的情况下，我们在新颖的框架中利用了可解释性方法来获取本地化地图。 即使在与ImageNet的域相距遥远的域中，我们也可以在各种任务和许多数据集上取得优异的结果。
