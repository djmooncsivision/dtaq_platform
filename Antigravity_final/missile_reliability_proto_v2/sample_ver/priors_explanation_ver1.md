# 베이지안 모델의 사전 분포(Prior Distribution) 상세 설명 (ver1)

베이지안 모델에서 **사전 분포(Prior Distribution)** 설정은 분석의 철학과 결과를 결정하는 매우 중요한 과정입니다. 사전 분포는 **"우리가 데이터를 보기 전에, 모델의 각 파라미터에 대해 무엇을 알고 있거나 믿고 있는가?"**를 수학적으로 표현한 것입니다.

이 스크립트의 `run_forest_plot_comparison.py` 함수에 사용된 사전 분포들은 다음과 같습니다.

---

### 1. 전체 평균 신뢰도: `mu_global_logit`

```python
mu_global_logit = pm.Normal('mu_global_logit', mu=3.89, sigma=0.5)
```

*   **역할**: 이 파라미터는 모든 생산 연도와 모든 LOT을 아우르는 **'전체 평균 신뢰도'**를 나타냅니다. 모델의 가장 근간이 되는 기준점입니다.
*   **`logit` 변환**: 신뢰도(확률)는 0과 1 사이의 값을 갖지만, 통계 모델은 제약 없는 실수 범위(-∞ ~ +∞)에서 작동할 때 더 안정적입니다. `logit` 함수는 확률을 실수 전체 범위로 변환해주는 역할을 합니다. (`logit(0.98) ≈ 3.89`)
*   **`mu=3.89`**: 이것이 바로 **"사전 지식(Prior Knowledge)"**의 핵심입니다. 우리는 이 제품의 평균 신뢰도가 약 98% (`invlogit(3.89) ≈ 0.98`)일 것이라는 강력한 믿음을 모델에 알려주는 것입니다.
*   **`sigma=0.5`**: 이 값은 우리의 믿음에 대한 **'불확실성'**을 나타냅니다. 표준편차를 0.5로 설정함으로써, "평균 신뢰도가 98% 근처일 것 같지만, 대략 95%에서 99.2% 사이의 값도 충분히 가능하다"고 말하는 것과 같습니다.

### 2. 연도 간 편차의 크기: `sigma_year`

```python
sigma_year = pm.HalfNormal('sigma_year', sigma=model_params["inter_year_sigma"])
```

*   **역할**: **'연도별 평균 신뢰도가 전체 평균에서 얼마나 벗어날 수 있는가'**를 제어하는 '하이퍼파라미터(hyperparameter)'입니다. 즉, 연도 간 품질 변동성의 크기를 결정합니다.
*   **`HalfNormal` 분포**: 표준편차는 항상 0보다 커야 하므로, 0 이상에서만 정의되는 `HalfNormal` 분포를 사용합니다.
*   **`sigma=model_params[...]`**: 이 부분이 바로 **"낙관적/보수적 시나리오"**를 구현하는 곳입니다.
    *   **낙관적 시나리오 (`sigma=0.01`)**: `sigma_year`가 0에 매우 가깝도록 강하게 제약하여 "생산 연도가 달라도 품질은 거의 균일할 것이다"라는 믿음을 주입합니다.
    *   **보수적 시나리오 (`sigma=0.2`)**: `sigma_year`가 더 큰 값을 가질 수 있도록 허용하여 "연도별로 품질 차이가 꽤 클 수도 있다"는 관점을 반영합니다.

### 3. LOT 간 편차의 크기: `sigma_lot_base`

```python
sigma_lot_base = pm.HalfNormal('sigma_lot_base', sigma=model_params["intra_lot_sigma"])
```

*   **역할**: **'같은 연도 내에서 각 LOT의 신뢰도가 해당 연도의 평균에서 얼마나 벗어날 수 있는가'**를 제어합니다. 즉, LOT 간 품질 변동성의 크기를 결정합니다.
*   **시나리오별 가정**:
    *   **낙관적 시나리오 (`sigma=0.02`)**: "같은 해에 만들어졌다면, LOT별 품질 차이는 거의 없을 것이다"라는 믿음을 반영합니다.
    *   **보수적 시나리오 (`sigma=0.1`)**: "같은 해에 만들어졌더라도 LOT별로 품질이 다를 수 있다"는 가능성을 더 크게 열어둡니다.

### 4. (보수적 시나리오) 경년열화 효과: `variance_degradation_rate`

```python
variance_degradation_rate = pm.HalfNormal('variance_degradation_rate', sigma=0.05)
age_of_lot = CURRENT_YEAR - indices["year_of_lot"]
sigma_lot_effective = pm.Deterministic('sigma_lot_effective', sigma_lot_base + age_of_lot * variance_degradation_rate)
```

*   **역할**: 보수적 시나리오에만 적용되며, **"오래된 LOT일수록 품질의 불확실성(편차)이 더 커질 것이다"**라는 가정을 모델링합니다.
*   `variance_degradation_rate`: '시간이 1년 지날 때마다 LOT의 신뢰도 편차가 얼마나 증가하는가'를 나타내는 비율입니다.
*   `sigma_lot_effective`: 최종적으로 각 LOT에 적용되는 편차입니다. 기본 LOT 편차(`sigma_lot_base`)에 LOT의 나이(`age_of_lot`)와 열화율을 곱한 값을 더해 계산됩니다. 이로 인해 오래된 LOT일수록 더 넓은 신뢰도 추정 범위를 갖게 됩니다.

---

이처럼 사전 분포 설정은 분석가가 가진 사전 지식, 가정, 그리고 탐구하고 싶은 시나리오를 모델에 명시적으로 알려주는 과정입니다. 이 가정이 데이터와 결합하여 최종 추론 결과인 **사후 분포(Posterior Distribution)**를 만들어내는 것이 베이지안 통계의 핵심 원리입니다.