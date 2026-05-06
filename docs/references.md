# Referencias — GeoQuant-MobileNet

> Referencias completas del proyecto **GeoQuant: El costo angular de la cuantización — Integridad geométrica vs. latencia INT8 en MobileNetV3**.

Este archivo organiza las referencias por eje temático para facilitar su consulta desde el repositorio y desde el QR del póster.

---

## Índice

- [1. Fundamentos de eficiencia, redes ligeras y MobileNet](#1-fundamentos-de-eficiencia-redes-ligeras-y-mobilenet)
- [2. Cuantización de redes neuronales: teoría, PTQ, QAT y despliegue](#2-cuantización-de-redes-neuronales-teoría-ptq-qat-y-despliegue)
- [3. Reconocimiento visual fino, embeddings y aprendizaje métrico](#3-reconocimiento-visual-fino-embeddings-y-aprendizaje-métrico)
- [4. Geometría, similitud representacional y métricas de evaluación](#4-geometría-similitud-representacional-y-métricas-de-evaluación)

---

## 1. Fundamentos de eficiencia, redes ligeras y MobileNet

### [1]

V. Sze, Y. H. Chen, T. J. Yang y J. S. Emer, **“Efficient Processing of Deep Neural Networks: A Tutorial and Survey,”** Proceedings of the IEEE, vol. 105, n.º 12, págs. 2295- 2329, dic. de 2017. doi: 10.1109/ JPROC.2017.2761740. dirección: https://ieeexplore.ieee.org/abstract/document/8114708.

### [5]

A. Howard et al., **“Searching for mobileNetV3,”** Proceedings of the IEEE International Conference on Computer Vision, págs. 1314 -1324, oct. de 2019. doi: 10 . 1109 / ICCV . 2019 . 00140. dirección: https://ieeexplore.ieee.org/document/9008835.

### [6]

J. Hu, L. Shen y G. Sun, **“Squeeze-and-Excitation Networks,”** Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, págs. 7132- 7141, dic. de 2018. doi: 10.1109/CVPR.2018.00745. dirección: https://ieeexplore.ieee.org/document/8578843.

### [21]

M. Sandler, A. Howard, M. Zhu, A. Zhmoginov y L. C. Chen, **“MobileNetV2: Inverted Residuals and Linear Bottlenecks,”** Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, págs. 4510- 4520, ene. de 2018. doi: 10.1109/CVPR.2018.00474. dirección: https://arxiv.org/pdf/1801.04381.

### [22]

P. Ramachandran, B. Zoph y Q. V. Le Google Brain, **“Searching for Activation Functions,”** 6th International Conference on Learning Representations, ICLR 2018 - Workshop Track Proceedings, oct. de 2017. dirección: https://arxiv.org/pdf/1710.05941.

### [28]

H.- I. Liu et al., **“Lightweight Deep Learning for Resource-Constrained Environments: A Survey,”** ACM Computing Surveys, vol. 56, n.º 10, págs. 1-42, 2024. doi: 10.1145/3657282.

---

## 2. Cuantización de redes neuronales: teoría, PTQ, QAT y despliegue

### [2]

B. Rokh, A. Azarpeyvand y A. Khanteymoori, **“A Comprehensive Survey on Model Quantization for Deep Neural Networks in Image Classification,”** ACM Transactions on Intelligent Systems and Technology, vol. 14, n.º 6, págs. 1-50, 2023. doi: 10.1145/3623402.

### [3]

B. Jacob et al., **“Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference,”** 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, págs. 2704- 2713, dic. de 2018. doi: 10.1109/CVPR.2018.00286.

### [4]

M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen y T. Blankevoort, **“A White Paper on Neural Network Quantization,”** arXiv preprint arXiv:2106.08295, abr. de 2021. doi: 10.48550/arXiv.2106.08295.

### [7]

S. Yun y A. Wong, **“Do All MobileNets Quantize Poorly? Gaining Insights into the Effect of Quantization on Depthwise Separable Convolutional Networks Through the Eyes of Multi-scale Distributional Dynamics,”** en Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, abr. de 2021, págs. 2447-2456. doi: 10.1109/CVPRW53098.2021.00277.

### [8]

M. Nagel, M. Fournarakis, Y. Bondarenko y T. Blankevoort, **“Overcoming Oscillations in Quantization- Aware Training,”** Proceedings of Machine Learning Research, vol. 162, págs. 16 318- 16 330, mar. de 2022. dirección: https://arxiv.org/pdf/2203.11086.

### [9]

E. Park y S. Yoo, **“PROFIT: A Novel Training Method for sub-4-bit MobileNet Models,”** Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), vol. 12351 LNCS, págs. 430- 446, ago. de 2020. dirección: http://arxiv.org/abs/ 2008.04693.

### [13]

Y. Bhalgat, J. Lee, M. Nagel, T. Blankevoort y N. Kwak, **“LSQ+: Improving low-bit quantization through learnable offsets and better initialization,”** IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops, vol. 2020-June, págs. 2978-2985, abr. de 2020. doi: 10. 1109/CVPRW50498.2020.00356. dirección: https://arxiv.org/pdf/2004.09576.

### [23]

M. Nagel, M. V. Baalen, T. Blankevoort y M. Welling, **“Data-Free Quantization Through Weight Equalization and Bias Correction,”** Proceedings of the IEEE International Conference on Computer Vision, págs. 1325-1334, jun. de 2019. doi: 10.1109/ICCV.2019.00141. dirección: https://arxiv. org/pdf/1906.04721.

### [24]

T. Han, D. Li, J. Liu, L. Tian e Y. Shan, **“Improving Low-Precision Network Quantization via Bin Regularization,”** IEEE International Conference on Computer Vision, págs. 5241- 5250, 2021. doi: 10.1109/ICCV48922.2021.00521.

### [25]

R. Zhao, Y. Hu, J. Dotzel, C. de Sa y Z. Zhang, **“Improving Neural Network Quantization without Retraining using Outlier Channel Splitting,”** 36th International Conference on Machine Learning, ICML 2019, vol. 2019-June, págs. 13 012 - 13 021, ene. de 2019. dirección: https://arxiv.org/pdf/1901.09504.

### [26]

X. Chang, **“Smoothed Per-tensor Weight Quantization: A Robust Solution for Neural Network Deploy- ment,”** International Journal of Electronics and Telecommunication, vol. 71, n.º 3, págs. 1 - 7, jul. de 2025. doi: 10.24425/ijet.2025.153629. dirección: https://ijet.ise.pw.edu.pl/index.php/ ijet/article/view/10.24425-ijet.2025.153629.

### [27]

R. M. Gray y D. L. Neuhoff, **“Quantization,”** IEEE Transactions on Information Theory, vol. 44, n.º 6, págs. 2325-2383, abr. de 1998. doi: 10.1109/18.720541.

### [29]

L. Wei, Z. Ma, C. Yang y Q. Yao, **“Advances in the Neural Network Quantization: A Comprehensive Review,”** Applied Sciences, vol. 14, n.º 17, pág. 7445, 2024. doi: 10.3390/app14177445.

### [30]

J. Zhang, Y. Zhou y R. Saab, **“Post-Training Quantization for Neural Networks with Provable Guaran- tees,”** SIAM Journal on Mathematics of Data Science, abr. de 2023. doi: 10.48550/arXiv.2201.11113.

### [31]

J. Lee, M. Yu, Y. Kwon y T. Kim, **“Quantune: Post-Training Quantization of Convolutional Neural Networks Using Extreme Gradient Boosting for Fast Deployment,”** Future Generation Computer Systems, vol. 132, págs. 124-135, abr. de 2022. doi: 10.1016/j.future.2022.02.005.

### [32]

M. Nagel, R. A. Amjad, M. van Baalen, C. Louizos y T. Blankevoort, **“Up or Down? Adaptive Rounding for Post-Training Quantization,”** Proceedings of the 37th International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research, vol. 119, págs. 7197- 7206, jun. de 2020. dirección: http://arxiv.org/abs/2004.10568.

---

## 3. Reconocimiento visual fino, embeddings y aprendizaje métrico

### [10]

J. Deng, J. Guo, J. Yang, N. Xue, I. Kotsia y S. Zafeiriou, **“ArcFace: Additive Angular Margin Loss for Deep Face Recognition,”** IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, n.º 10, págs. 5962-5979, 2022. doi: 10.1109/TPAMI.2021.3087709.

### [11]

C. Wah, S. Branson, P. Welinder, P. Perona y S. Belongie, **“The Caltech-UCSD Birds-200-2011 Dataset,”** California Institute of Technology, inf. téc. CNS-TR-2011-001, 2011.

### [12]

X.- S. Wei et al., **“Fine-Grained Image Analysis with Deep Learning: A Survey,”** IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, n.º 12, págs. 8927 - 8948, 2022. doi: 10.1109/TPAMI. 2021.3126648.

### [33]

Y. Bengio, A. Courville y P. Vincent, **“Representation Learning: A Review and New Perspectives,”** IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 35, n.º 8, págs. 1798 - 1828, abr. de 2013. doi: 10.1109/TPAMI.2013.50.

### [34]

D. Li e Y. Tian, **“Survey and Experimental Study on Metric Learning Methods,”** Neural Networks, vol. 105, págs. 447-462, abr. de 2018. doi: 10.1016/j.neunet.2018.06.003.

### [35]

W. Liu, Y. Wen, B. Raj, R. Singh y A. Weller, **“SphereFace Revived: Unifying Hyperspherical Fa- ce Recognition,”** IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, n.º 2, págs. 2458-2474, 2023. doi: 10.1109/TPAMI.2022.3159732.

---

## 4. Geometría, similitud representacional y métricas de evaluación

### [14]

S. Kornblith, M. Norouzi, H. Lee y G. Hinton, **“Similarity of Neural Network Representations Revisited,”** en Proceedings of the 36th International Conference on Machine Learning (ICML), ép. Proceedings of Machine Learning Research, vol. 97, PMLR, 2019, págs. 3519-3529.

### [15]

T. Wang y P. Isola, **“Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere,”** 37th International Conference on Machine Learning, ICML 2020, vol. PartF168147-13, págs. 9871- 9881, mayo de 2020. dirección: https://arxiv.org/pdf/2005.10242.

### [16]

K. Ethayarajh, **“How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings,”** en Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Association for Computational Linguistics, 2019, págs. 55 - 65. doi: 10.18653/v1/D19-1006.

### [17]

J. Venna y S. Kaski, **“Local Multidimensional Scaling,”** Neural Networks, vol. 19, n.º 6-7, págs. 889- 899, 2006. doi: 10.1016/j.neunet.2006.05.014.

### [18]

O. Roy y M. Vetterli, **“The Effective Rank: A Measure of Effective Dimensionality,”** en 15th European Signal Processing Conference (EUSIPCO), 2007, págs. 606-610.

### [19]

M. Klabunde, T. Schumacher, M. Strohmaier y F. Lemmerich, **“Similarity of Neural Network Models: A Survey of Functional and Representational Measures,”** arXiv preprint arXiv:2305.06329, 2023. doi: 10.48550/arXiv.2305.06329.

### [20]

M. Klabunde, T. Schumacher, M. Strohmaier y F. Lemmerich, **“ReSi: A Comprehensive Benchmark for Representational Similarity Measures,”** arXiv preprint arXiv:2408.00531, 2024. doi: 10.48550/arXiv. 2408.00531.

---

## Lista completa en orden numérico

**[1]** V. Sze, Y. H. Chen, T. J. Yang y J. S. Emer, **“Efficient Processing of Deep Neural Networks: A Tutorial and Survey,”** Proceedings of the IEEE, vol. 105, n.º 12, págs. 2295- 2329, dic. de 2017. doi: 10.1109/ JPROC.2017.2761740. dirección: https://ieeexplore.ieee.org/abstract/document/8114708.

**[2]** B. Rokh, A. Azarpeyvand y A. Khanteymoori, **“A Comprehensive Survey on Model Quantization for Deep Neural Networks in Image Classification,”** ACM Transactions on Intelligent Systems and Technology, vol. 14, n.º 6, págs. 1-50, 2023. doi: 10.1145/3623402.

**[3]** B. Jacob et al., **“Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference,”** 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, págs. 2704- 2713, dic. de 2018. doi: 10.1109/CVPR.2018.00286.

**[4]** M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen y T. Blankevoort, **“A White Paper on Neural Network Quantization,”** arXiv preprint arXiv:2106.08295, abr. de 2021. doi: 10.48550/arXiv.2106.08295.

**[5]** A. Howard et al., **“Searching for mobileNetV3,”** Proceedings of the IEEE International Conference on Computer Vision, págs. 1314 -1324, oct. de 2019. doi: 10 . 1109 / ICCV . 2019 . 00140. dirección: https://ieeexplore.ieee.org/document/9008835.

**[6]** J. Hu, L. Shen y G. Sun, **“Squeeze-and-Excitation Networks,”** Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, págs. 7132- 7141, dic. de 2018. doi: 10.1109/CVPR.2018.00745. dirección: https://ieeexplore.ieee.org/document/8578843.

**[7]** S. Yun y A. Wong, **“Do All MobileNets Quantize Poorly? Gaining Insights into the Effect of Quantization on Depthwise Separable Convolutional Networks Through the Eyes of Multi-scale Distributional Dynamics,”** en Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, abr. de 2021, págs. 2447-2456. doi: 10.1109/CVPRW53098.2021.00277.

**[8]** M. Nagel, M. Fournarakis, Y. Bondarenko y T. Blankevoort, **“Overcoming Oscillations in Quantization- Aware Training,”** Proceedings of Machine Learning Research, vol. 162, págs. 16 318- 16 330, mar. de 2022. dirección: https://arxiv.org/pdf/2203.11086.

**[9]** E. Park y S. Yoo, **“PROFIT: A Novel Training Method for sub-4-bit MobileNet Models,”** Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), vol. 12351 LNCS, págs. 430- 446, ago. de 2020. dirección: http://arxiv.org/abs/ 2008.04693.

**[10]** J. Deng, J. Guo, J. Yang, N. Xue, I. Kotsia y S. Zafeiriou, **“ArcFace: Additive Angular Margin Loss for Deep Face Recognition,”** IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, n.º 10, págs. 5962-5979, 2022. doi: 10.1109/TPAMI.2021.3087709.

**[11]** C. Wah, S. Branson, P. Welinder, P. Perona y S. Belongie, **“The Caltech-UCSD Birds-200-2011 Dataset,”** California Institute of Technology, inf. téc. CNS-TR-2011-001, 2011.

**[12]** X.- S. Wei et al., **“Fine-Grained Image Analysis with Deep Learning: A Survey,”** IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, n.º 12, págs. 8927 - 8948, 2022. doi: 10.1109/TPAMI. 2021.3126648.

**[13]** Y. Bhalgat, J. Lee, M. Nagel, T. Blankevoort y N. Kwak, **“LSQ+: Improving low-bit quantization through learnable offsets and better initialization,”** IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops, vol. 2020-June, págs. 2978-2985, abr. de 2020. doi: 10. 1109/CVPRW50498.2020.00356. dirección: https://arxiv.org/pdf/2004.09576.

**[14]** S. Kornblith, M. Norouzi, H. Lee y G. Hinton, **“Similarity of Neural Network Representations Revisited,”** en Proceedings of the 36th International Conference on Machine Learning (ICML), ép. Proceedings of Machine Learning Research, vol. 97, PMLR, 2019, págs. 3519-3529.

**[15]** T. Wang y P. Isola, **“Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere,”** 37th International Conference on Machine Learning, ICML 2020, vol. PartF168147-13, págs. 9871- 9881, mayo de 2020. dirección: https://arxiv.org/pdf/2005.10242.

**[16]** K. Ethayarajh, **“How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings,”** en Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Association for Computational Linguistics, 2019, págs. 55 - 65. doi: 10.18653/v1/D19-1006.

**[17]** J. Venna y S. Kaski, **“Local Multidimensional Scaling,”** Neural Networks, vol. 19, n.º 6-7, págs. 889- 899, 2006. doi: 10.1016/j.neunet.2006.05.014.

**[18]** O. Roy y M. Vetterli, **“The Effective Rank: A Measure of Effective Dimensionality,”** en 15th European Signal Processing Conference (EUSIPCO), 2007, págs. 606-610.

**[19]** M. Klabunde, T. Schumacher, M. Strohmaier y F. Lemmerich, **“Similarity of Neural Network Models: A Survey of Functional and Representational Measures,”** arXiv preprint arXiv:2305.06329, 2023. doi: 10.48550/arXiv.2305.06329.

**[20]** M. Klabunde, T. Schumacher, M. Strohmaier y F. Lemmerich, **“ReSi: A Comprehensive Benchmark for Representational Similarity Measures,”** arXiv preprint arXiv:2408.00531, 2024. doi: 10.48550/arXiv. 2408.00531.

**[21]** M. Sandler, A. Howard, M. Zhu, A. Zhmoginov y L. C. Chen, **“MobileNetV2: Inverted Residuals and Linear Bottlenecks,”** Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, págs. 4510- 4520, ene. de 2018. doi: 10.1109/CVPR.2018.00474. dirección: https://arxiv.org/pdf/1801.04381.

**[22]** P. Ramachandran, B. Zoph y Q. V. Le Google Brain, **“Searching for Activation Functions,”** 6th International Conference on Learning Representations, ICLR 2018 - Workshop Track Proceedings, oct. de 2017. dirección: https://arxiv.org/pdf/1710.05941.

**[23]** M. Nagel, M. V. Baalen, T. Blankevoort y M. Welling, **“Data-Free Quantization Through Weight Equalization and Bias Correction,”** Proceedings of the IEEE International Conference on Computer Vision, págs. 1325-1334, jun. de 2019. doi: 10.1109/ICCV.2019.00141. dirección: https://arxiv. org/pdf/1906.04721.

**[24]** T. Han, D. Li, J. Liu, L. Tian e Y. Shan, **“Improving Low-Precision Network Quantization via Bin Regularization,”** IEEE International Conference on Computer Vision, págs. 5241- 5250, 2021. doi: 10.1109/ICCV48922.2021.00521.

**[25]** R. Zhao, Y. Hu, J. Dotzel, C. de Sa y Z. Zhang, **“Improving Neural Network Quantization without Retraining using Outlier Channel Splitting,”** 36th International Conference on Machine Learning, ICML 2019, vol. 2019-June, págs. 13 012 - 13 021, ene. de 2019. dirección: https://arxiv.org/pdf/1901.09504.

**[26]** X. Chang, **“Smoothed Per-tensor Weight Quantization: A Robust Solution for Neural Network Deploy- ment,”** International Journal of Electronics and Telecommunication, vol. 71, n.º 3, págs. 1 - 7, jul. de 2025. doi: 10.24425/ijet.2025.153629. dirección: https://ijet.ise.pw.edu.pl/index.php/ ijet/article/view/10.24425-ijet.2025.153629.

**[27]** R. M. Gray y D. L. Neuhoff, **“Quantization,”** IEEE Transactions on Information Theory, vol. 44, n.º 6, págs. 2325-2383, abr. de 1998. doi: 10.1109/18.720541.

**[28]** H.- I. Liu et al., **“Lightweight Deep Learning for Resource-Constrained Environments: A Survey,”** ACM Computing Surveys, vol. 56, n.º 10, págs. 1-42, 2024. doi: 10.1145/3657282.

**[29]** L. Wei, Z. Ma, C. Yang y Q. Yao, **“Advances in the Neural Network Quantization: A Comprehensive Review,”** Applied Sciences, vol. 14, n.º 17, pág. 7445, 2024. doi: 10.3390/app14177445.

**[30]** J. Zhang, Y. Zhou y R. Saab, **“Post-Training Quantization for Neural Networks with Provable Guaran- tees,”** SIAM Journal on Mathematics of Data Science, abr. de 2023. doi: 10.48550/arXiv.2201.11113.

**[31]** J. Lee, M. Yu, Y. Kwon y T. Kim, **“Quantune: Post-Training Quantization of Convolutional Neural Networks Using Extreme Gradient Boosting for Fast Deployment,”** Future Generation Computer Systems, vol. 132, págs. 124-135, abr. de 2022. doi: 10.1016/j.future.2022.02.005.

**[32]** M. Nagel, R. A. Amjad, M. van Baalen, C. Louizos y T. Blankevoort, **“Up or Down? Adaptive Rounding for Post-Training Quantization,”** Proceedings of the 37th International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research, vol. 119, págs. 7197- 7206, jun. de 2020. dirección: http://arxiv.org/abs/2004.10568.

**[33]** Y. Bengio, A. Courville y P. Vincent, **“Representation Learning: A Review and New Perspectives,”** IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 35, n.º 8, págs. 1798 - 1828, abr. de 2013. doi: 10.1109/TPAMI.2013.50.

**[34]** D. Li e Y. Tian, **“Survey and Experimental Study on Metric Learning Methods,”** Neural Networks, vol. 105, págs. 447-462, abr. de 2018. doi: 10.1016/j.neunet.2018.06.003.

**[35]** W. Liu, Y. Wen, B. Raj, R. Singh y A. Weller, **“SphereFace Revived: Unifying Hyperspherical Fa- ce Recognition,”** IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, n.º 2, págs. 2458-2474, 2023. doi: 10.1109/TPAMI.2022.3159732.
