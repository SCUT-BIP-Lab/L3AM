# L3AM: Linear Adaptive Additive Angular Margin Loss for Video-based Hand Gesture Authentication
Pytorch Implementation of paper:

> **L3AM: Linear Adaptive Additive Angular Margin Loss for Video-based Hand Gesture Authentication**
>
> Wenwei Song, Wenxiong Kang\*, Adams Wai-Kin Kong\*, Yufeng Zhang and Yitao Qiao.

## Main Contribution
Feature extractors significantly impact the performance of biometric systems. In the field of hand gesture authentication, existing studies focus on improving the model architectures and behavioral characteristic representation methods to enhance their feature extractors. However, loss functions, which can guide extractors to produce more discriminative identity features, are neglected. In this paper, we improve the margin-based Softmax loss functions, which are mainly designed for face authentication, in two aspects to form a new loss function for hand gesture authentication. First, we propose to replace the commonly used cosine function in the margin-based Softmax losses with a linear function to measure the similarity between identity features and proxies (the weight matrix of Softmax, which can be viewed as class centers). With the linear function, the main gradient magnitude decreases monotonically as the quality of the model improves during training, thus allowing the model to be quickly optimized in the early stage and precisely fine-tuned in the late stage. Second, we design an adaptive margin scheme to assign margin penalties to different samples according to their separability and the model quality in each iteration. Our adaptive margin scheme constrains the gradient magnitude. It can reduce radical (excessively large) gradient magnitudes and provide moderate (not too small) gradient magnitudes for model optimization, contributing to more stable training. The linear function and the adaptive margin scheme are complementary. Combining them, we obtain the proposed Linear Adaptive Additive Angular Margin (L3AM) loss. To demonstrate the effectiveness of L3AM loss, we conduct extensive experiments on seven hand-related authentication datasets, compare it with 25 state-of-the-art (SOTA) loss functions, and apply it to eight SOTA hand gesture authentication models. The experimental results show that L3AM loss further improves the performance of the eight authentication models and outperforms the 25 losses.



## Comparisons with ArcFace-based Losses
ArcFace is one of the most popular loss functions currently, and it has achieved SOTA performance on ten face recognition benchmarks, including video datasets. Thus, many enhanced loss functions are proposed based on ArcFace by designing more advanced margin penalties (e.g., Dyn-arcFace, ElasticFace-Arc, CurricularFace, MV-Softmax , and MagFace) or regularization terms (e.g., AdaptiveFace, RegularFace, MagFace, IHEM, and Regss). In this section, we implement these loss functions on DwTNL-Net and test them on SCUT-DHGA dataset under MG protocol. From the Figure , we find that Dyn-arcFace, CurricularFace, AdaptiveFace, and IHEM are effective on the SCUT-DHGA dataset, which can reduce the EER of ArcFace from 0.652\% to 0.641\%, 0.626\%, 0.648\%, and 0.634\%, respectively. However, these loss functions are still inferior to our L3AM loss (0.611\%) with a fixed margin penalty (L3AM(m=0.4)), demonstrating the effectiveness of the deduced linear similarity measurement function. We also integrate ArcFace with the proposed adaptive margin scheme and obtain ArcFace*. The EER of ArcFace is further decreased to 0.585\%, which is lower than the four effective improved losses mentioned above and shows that the proposed adaptive margin scheme is also applicable to ArcFace. Finally, our L3AM loss obtains the lowest EER (0.559\%) among these ArcFace-based losses.


 <div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/L3AM/main/img/L3AM_Arc." width="600" />
 </p>
 </div>  
 Comparisons with ArcFace-based losses on SCUT-DHGA dataset under MG protocol. ArcFace* is the combination of ArcFace and our adaptive margin scheme. L3AM(m=0.4) denotes L3AM loss with a fixed margin penalty $m=0.4$.

## Comparisons with Other Losses

There are also a large number of SOTA losses that are not designed based on ArcFace (e.g., CenterLoss, NormFace, AdaCos, EqMLoss, CVMLoss, CosFace, SphereFace, SphereFace+, and SphereFaceR) or even not based on Softmax loss (e.g., SFace, CircleLoss, SphereFace2, and DSoftmax). In this section, we employ these loss functions on DwTNL-Net and test them on SCUT-DHGA dataset under MG protocol. From the Figure, we can find that the performances of these non-Softmax losses are not very effective on SCUT-DHGA dataset. Hence, we chose to design the loss function based on Softmax for hand gesture authentication at the beginning of our work. Vanilla Softmax and CenterLoss do not employ normalization and rescaling operations, and they underperform NormFace and NormFace-based losses (e.g., CosFace, SphereFace, and our L3AM loss), confirming the significance of normalization and rescaling. In addition, some loss functions (e.g., NormFace, AdaCos, SFace, and DSoftmax) are designed without margin penalty, and their performances are not very high. Thus, the introduction of margin penalties is helpful for model optimization.CosFace and SphereFace are another two well-known loss functions besides ArcFace. They are stable and can obtain decent EERs on SCUT-DHGA dataset. CosFace, also known as AMSoftmax, is the loss function used in the current SOTA hand gesture authentication models. CVMLoss addresses the class imbalance and Softmax saturation issues by introducing a true-class margin penalty and a false-class margin penalty to enhance CosFace. Experimental results show that this strategy can reduce the EER of CosFace from 0.689\% to 0.659\%. SphereFace+ designed a minimum hyperspherical energy regularization term for SphereFace to promote the diversity of neurons. SphereFaceR redesigned a target angular function and proposed a characteristic gradient detachment strategy to improve the training stability and generalizability. As shown in the Figure, SphereFace+ and SphereFaceR reduce the EER of SphereFace from 0.704\% to 0.678\% and 0.641\%, respectively, and both outperform CosFace. However, these loss functions do not perform as well as our L3AM loss in hand gesture authentication. Finally, our L3AM loss obtains an EER of 0.559\%, which is 0.13\% lower than the commonly used CosFace in hand gesture authentication. 


 <div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/L3AM/main/img/L3AM_Other." width="600" />
 </p>
 </div>  
Comparisons with other SOTA losses on SCUT-DHGA dataset under MG protocol. CosFace (also known as AMSoftmax) is the loss function employed in the current SOTA models for hand gesture authentication. 



## Dependencies
Please make sure the following libraries are installed successfully:
- [PyTorch](https://pytorch.org/) >= 1.7.0

## Hand gesture authentication

If you want to explore the entire dynamic hand gesture authentication framework, please refer to our pervious work [SCUT-DHGA](https://github.com/SCUT-BIP-Lab/SCUT-DHGA) 
or send an email to Prof. Kang (auwxkang@scut.edu.cn).
