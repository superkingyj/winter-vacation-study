# Optimization

**딥러닝**은 깊은 신경망 계층을 말합니다.

입력층과 출력층 사이에 여러 개의 은닉층으로 이루어진 신경망입니다.

층이 깊어질수록 모듈과 함수에 따른 **하이퍼파라미터(hyper-parameter)**도 비례하여 많아집니다.

이 하이퍼파라미터를 결정하여 모델이 정확하게 결과를 뱉어낼 수 있도록 하는 것이 학습의 핵심입니다.

그러기 위해서는 Loss Function을 정의하여야 합니다. 

그리고 **Loss Function을 Optimize** 해야합니다. 

다음 그림에 설명이 잘 되어있듯이,

모델은 Loss Function**(틀린 정도)**을 가지고 있습니다.

고등학교 때 수학시간에 배웠던 기억을 되살려봅니다.

미분하여 그 점에서의 미분 값이

-   (+) 값이 나오면 증가
-   (-) 값이 나오면 감소

마찬가지로 결국 loss function도 어떠한 함수의 형태를 띄고있고

loss를 줄여가는 방향으로 가면 되니까

감소하는 방향, 미분값이 (-)가 되는 방향으로 가면 됩니다.  

그리고 한번에 얼마만큼씩 움직일 것인지가 learning rate입니다.

빨리 학습하고 싶어서 크게 잡으면 발산할 수 있기 때문에 적절한 한발자국을 찾아야 합니다.

[##_Image|kage@dxizRz/btqBhEhgY5M/96KPlpMCSefnCaqt5W8mx0/img.jpg|alignCenter|data-filename="-35-638.jpg" data-origin-width="638" data-origin-height="359"|출처: 하용호 - 자습해도 모르겠던 딥러닝 인스톨시켜드립니다.||_##]

## **SGD**

앞서 살펴보았듯, 최적의 가중치 값을 구하기 위해 미분을 통해 기울기를 구하고 가중치를 갱신하는 법을 배웠습니다. 그것이 바로 SGD입니다. 인공신경망을 학습시키기 위해서 모델의 예측값과 정답(라벨) 사이의 차이를 측정하기 위해 손실 함수를 정의했습니다.

Gradient Descent는 다음 수식 한 줄로 표현 됩니다.

[##_Image|kage@edgK1Z/btqBc50ZAwT/lhp03vZeqVuREioJOnxLkk/img.png|alignCenter|data-filename="0_pPP74czZRRfFe7y1.png" data-origin-width="394" data-origin-height="70"|||_##]

위에서 보았듯, 세터 값을 갱신하는데 세터에 따라 변화하는 Loss Function의 변화량을 낮추는 방향으로 하는 것입니다.

(앞의 거꾸로된 세모는 델이라고 하는 미분을 나타내는 기호입니다. )

[##_Image|kage@bAy1TN/btqBf29CS9J/Jtr8a685PrmEnExLSt55M0/img.png|alignCenter|data-filename="1_vYrcQ9UjJu_jUOeCsaD8YQ.png" data-origin-width="682" data-origin-height="209"|||_##]

하지만 Gradient Descent Optimize의 이동량은 미분계수에 의해 결정되는데,

미분계수가 0이 된다면 더 이상 업데이트되지 않습니다. 

왼쪽 그림과 같이 극값이 두개 이상인 경우 더이상 업데이트가 되지 않으면 Global minima를 찾을 수 없고, 안장점(saddle points)에 앉아버려 더 움직이지 못하는 경우도 있습니다. 

따라서 다음 방법이 나왔습니다. 

## **SGD with Momentum**

Gradient Descent Optimizer보다 개선된 알고리즘으로 **이동 값에 관성으로 인한 업데이트가 추가**된 Otimizer입니다. 

(로우가 0인 경우 SGD와 동일한 알고리즘이 됩니다.)

그렇지 않은 경우 vt만큼 원래의 방향으로 이동합니다. (관성)

[##_Image|kage@dyOMEZ/btqBfsOGIJP/qD7TBye7P5RYfljZ8YvDhk/img.png|alignCenter|data-filename="1_LjVeDQEHZBKC6C0TiUngWg.png" data-origin-width="709" data-origin-height="228" width="732" height="236"|||_##]

[##_Image|kage@cyQJcM/btqBfrPRec6/zxdnCp2Q7DoypsRjA9Kc6k/img.png|alignCenter|data-filename="momentum.png" data-origin-width="1228" data-origin-height="821" width="784" height="524"|출처: 알기 쉬운 산업수학||_##]

**[Momentum 적용해본 간단한 Tensorflow 코드](https://github.com/uyeonH/Tensorflow2.0-Tutorial/tree/master/Optimizer)**

[

uyeonH/Tensorflow2.0-Tutorial

Contribute to uyeonH/Tensorflow2.0-Tutorial development by creating an account on GitHub.

github.com



](https://github.com/uyeonH/Tensorflow2.0-Tutorial)

## **AdaGrad**

**\= Adaptive Gradient**

>   
> "지금까지 **많이 변화하지 않은 변수**들은 **step size를 크게** 하고,  
>   
> 지금까지 **많이 변화했던 변수**들은 **step size를 작게**하자"  
>   

자주 등장하거나 변화를 많이 했으면 optimum에 가까이 있을 확률이 높기 때문에

조금씩 이동하며 세밀하게 값을 조정합니다. 

반면, 적게 변화한 변수들은 optimum에 도달하기 위해 많이 이동해야 합니다.

따라서 빠르게 loss를 줄이는 방향으로 이동하려는 것입니다. 

AdaGrad는 이런 문제를 **Learning rate decay**로 해결합니다.

AdaGrad는 무한히 학습하다 보면 어느 순간 G가 너무 커져서 학습되지 않을 수 있는데 이를 RMSProp에서 해결합니다.

[##_Image|kage@cd4I2O/btqBg89HR5z/VCEKaKp4peD1n4WapnNVrk/img.png|alignCenter|data-filename="aa.PNG" data-origin-width="444" data-origin-height="168"|AdaGrad||_##]

[##_Image|kage@bLhmkW/btqBf4fNVM5/maOE0XTnnM4Up82qLxSjoK/img.png|alignCenter|data-filename="1_JgdKln9Iwx1IXWL4EwMFOw.png" data-origin-width="754" data-origin-height="250"|||_##]

## **RMSProp**

제프리 힌톤이 제안한 방법으로서, Adagrad의 단점을 해결하기 위한 방법입니다.

Adagrad의 식에서 gradient의 제곱 값을 더해나가면서

구한 Gt 부분을 합이 아니라 **지수 평균**으로 바꾸어서 대체한 방법입니다.

그러면 Adagrad처럼 Gt가 무한정 커지지는 않으면서 최근 변화량의 변수간 상대적인 크기 차이는 유지할 수 있습니다.

## **Adam Optimizer **

Adam(Adaptive Moment Estimation)은 RMSProp과 Momentum 방식을 합친 것 같은 알고리즘입니다.

아담의 강점은 **bounded step size**입니다.

이 방식에서는

Momentum 방식과 유사하게 **지금까지 계산해온 기울기의 지수 평균을 저장**하며,

RMSProp과 유사하게 **기울기의 제곱값의 지수평균을 저장**합니다.

[##_Image|kage@b6qN6N/btqBg9AMlfl/Sd2kCIWtFwqZRPbXug5V51/img.png|alignCenter|data-filename="ssdsd.PNG" data-origin-width="381" data-origin-height="117"|||_##]

다만, Adam에서는 m과 v가 처음에 0으로 초기화되어 있기 때문에

학습의 초반부에서는 mt,vtmt,vt가 0에 가깝게 bias 되어있을 것이라고 판단하여

이를 unbiased 하게 만들어주는 작업을 거칩니다. 

mt 와 vt의 식을 ∑ 형태로 펼친 후 양변에 expectation을 씌워서 정리해보면,

다음과 같은 보정을 통해 unbiased 된 expectation을 얻을 수 있습니다.

이 보정된 expectation들을 가지고 gradient가 들어갈 자리에 mt^, Gt가 들어갈 자리에 vt^를 넣어 계산을 진행합니다.

[##_Image|kage@bkRe49/btqBgDITtjS/KDKAlOXKM1yQ8XkEcejgKK/img.png|alignCenter|data-filename="df.PNG" data-origin-width="259" data-origin-height="239"|||_##]

[##_Image|kage@6hfca/btqBeLfwfbo/M8nXgyPE8Cp0JYF5vMgwWK/img.gif|alignCenter|data-filename="2dKCQHh.gif" data-origin-width="620" data-origin-height="480" width="658" height="509"|||_##]

## **결론**

최적화 방법과 하이퍼파라미터 값을 살펴보면 자신이 해결해야할 문제에 맞게 네트워크를 최적화하기 위한 직관력을 기를 수 있습니다. 하이퍼파라미터를 조사하는 동안 학습률, 배치사이즈, 옵티마이저 등의 최적화의 섬세함(sensitivity)을 직관적으로 이해하는 것이 좋습니다. 그 직관적인 이해가 좋은 모델을 만드는 데에 도움을 줄 것입니다.

### **Reference**

[https://www.deeplearning.ai/ai-notes/optimization/](https://www.deeplearning.ai/ai-notes/optimization/)

[

AI Notes: Parameter optimization in neural networks - deeplearning.ai

AI Notes: Parameter optimization in neural networks - deeplearning.ai

www.deeplearning.ai



](https://www.deeplearning.ai/ai-notes/optimization/)

[https://towardsdatascience.com/optimization-algorithms-in-deep-learning-191bfc2737a4](https://towardsdatascience.com/optimization-algorithms-in-deep-learning-191bfc2737a4)

[

Optimization Algorithms in Deep Learning

AdaGrad, RMSProp, Gradient Descent with Momentum & Adam Optimizer demystified

towardsdatascience.com



](https://towardsdatascience.com/optimization-algorithms-in-deep-learning-191bfc2737a4)

[https://icim.nims.re.kr/post/easyMath/428](https://icim.nims.re.kr/post/easyMath/428)

[

Momentum Optimizer | 알기 쉬운 산업수학 | 산업수학혁신센터

icim.nims.re.kr



](https://icim.nims.re.kr/post/easyMath/428)

[https://www.slideshare.net/yongho/ss-79607172?ref=https://mangastorytelling.tistory.com/entry/%ED%95%98%EC%9A%A9%ED%98%B8-%EC%9E%90%EC%8A%B5%ED%95%B4%EB%8F%84-%EB%AA%A8%EB%A5%B4%EA%B2%A0%EB%8D%98-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%A8%B8%EB%A6%AC%EC%86%8D%EC%97%90-%EC%9D%B8%EC%8A%A4%ED%86%A8-%EC%8B%9C%EC%BC%9C%EB%93%9C%EB%A6%BD%EB%8B%88%EB%8B%A4](https://www.slideshare.net/yongho/ss-79607172?ref=https://mangastorytelling.tistory.com/entry/%ED%95%98%EC%9A%A9%ED%98%B8-%EC%9E%90%EC%8A%B5%ED%95%B4%EB%8F%84-%EB%AA%A8%EB%A5%B4%EA%B2%A0%EB%8D%98-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%A8%B8%EB%A6%AC%EC%86%8D%EC%97%90-%EC%9D%B8%EC%8A%A4%ED%86%A8-%EC%8B%9C%EC%BC%9C%EB%93%9C%EB%A6%BD%EB%8B%88%EB%8B%A4)

[

자습해도 모르겠던 딥러닝, 머리속에 인스톨 시켜드립니다.

백날 자습해도 이해 안 가던 딥러닝, 머리속에 인스톨 시켜드립니다. 이 슬라이드를 보고 나면, 유명한 영상인식을 위한 딥러닝 구조 VGG를 코드 수준에서 읽으실 수 있을 거에요

www.slideshare.net



](https://www.slideshare.net/yongho/ss-79607172)
