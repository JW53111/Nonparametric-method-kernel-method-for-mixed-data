import numpy as np
from sklearn.metrics import accuracy_score, classification_report
# 从你的 code.py 导入模型类和数据加载函数
from code import MixedWeightedKernelClassifier, load_har_mixed_dataset

def evaluate_on_test():
    # 1. 设定参数（需与训练时保持一致）
    p_cts = 20  
    n_train = 500
    n_test = 300
    
    # 2. 加载数据
    print("正在加载数据...")
    X_train, y_train = load_har_mixed_dataset(split="train", n_samples=n_train, p_cts=p_cts)
    X_test, y_test = load_har_mixed_dataset(split="test", n_samples=n_test, p_cts=p_cts)
    
    # 3. 初始化并训练模型
    # 注意：这里会运行你在 code.py 里的优化算法
    model = MixedWeightedKernelClassifier(p_fun=9, p_cat=1, p_cts=p_cts)
    model.fit(X_train, y_train)
    
    # 4. 在测试集上预测
    print(f"\n正在对 {n_test} 个测试样本进行推理...")
    y_hat_continuous = model.predict(X_train, y_train, X_test)
    
    # 5. 转换标签：将回归值转为 1-6 的动作分类标签
    y_pred = np.round(y_hat_continuous).astype(int)
    y_pred = np.clip(y_pred, 1, 6)
    
    # 6. 输出论文所需的精确度结果
    acc = accuracy_score(y_test, y_pred)
    print("-" * 30)
    print(f"Eq.(22) 模型测试集精确度: {acc:.4f}")
    print("-" * 30)
    
    # 打印详细分类报告
    target_names = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    evaluate_on_test()