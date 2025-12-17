fig, axes = plt.subplots(1, 2, figsize=(16, 12), sharey=True)

for i, (name, trace) in enumerate(traces.items()):
    ax = axes[i]
    az.plot_forest(
        trace, 
        var_names=['reliability_lot'], 
        combined=True, 
        hdi_prob=0.9, # 90% 신뢰구간(HDI)
        ax=ax
    )
    ax.set_title(f'시나리오: {name}', fontsize=16)
    # y축 레이블은 첫 번째 그래프에만 표시
    if i == 0:
        ax.set_yticklabels(all_lots[::-1])
    else:
        ax.set_yticklabels([])

fig.suptitle('가정 시나리오별 롯트 신뢰도 추정 비교 (90% HDI)', fontsize=20, y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
