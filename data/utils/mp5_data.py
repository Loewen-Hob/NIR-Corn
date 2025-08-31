import scipy.io
import numpy as np
import pandas as pd

# -------------------------------
# 1. 加载 .mat 文件
# -------------------------------
data = scipy.io.loadmat('../corn.mat', struct_as_record=False, squeeze_me=True)

print("Available variables in mat file:")
for key in data.keys():
    if not key.startswith('__'):
        print(f"  {key}: {type(data[key])}, shape: {getattr(data[key], 'shape', 'unknown')}")

# -------------------------------
# 2. 提取 mp5spec（光谱数据）
# -------------------------------
if 'mp5spec' not in data:
    raise ValueError("❌ 'mp5spec' not found in the .mat file!")

spec_raw = data['mp5spec']

# 处理可能的嵌套结构（虽然通常是 ndarray）
if hasattr(spec_raw, 'data'):
    spec_values = spec_raw.data
else:
    spec_values = np.array(spec_raw)

if not isinstance(spec_values, np.ndarray):
    raise ValueError("❌ mp5spec is not a valid array!")

# 确保是 (80, 700) 形状
if len(spec_values.shape) == 1:
    spec_values = spec_values.reshape(-1, 700)  # 假设 80×700 展平成 1D
elif len(spec_values.shape) > 2:
    spec_values = spec_values.reshape(spec_values.shape[0], -1)

print(f"✅ mp5spec shape: {spec_values.shape}")

# -------------------------------
# 3. 提取 propvals（属性值）
# -------------------------------
if 'propvals' not in data:
    raise ValueError("❌ 'propvals' not found in the .mat file!")

prop_raw = data['propvals']
if hasattr(prop_raw, 'data'):
    prop_values = prop_raw.data
else:
    prop_values = np.array(prop_raw)

if not isinstance(prop_values, np.ndarray):
    raise ValueError("❌ propvals is not a valid array!")

# 确保是二维
if len(prop_values.shape) == 1:
    prop_values = prop_values.reshape(-1, 4)  # 假设 80×4
elif prop_values.shape[1] != 4:
    prop_values = prop_values[:, :4]  # 取前4列

print(f"✅ propvals shape: {prop_values.shape}")

# 检查样本数量是否匹配
n_samples_spec = spec_values.shape[0]
n_samples_prop = prop_values.shape[0]

if n_samples_spec != n_samples_prop:
    raise ValueError(f"❌ Sample count mismatch: mp5spec has {n_samples_spec}, propvals has {n_samples_prop}")

# -------------------------------
# 4. 创建 DataFrame
# -------------------------------
# 生成波长列名
wavelength_cols = [f'Wave_{i+1}' for i in range(spec_values.shape[1])]

# 创建光谱部分 DataFrame
df_spectra = pd.DataFrame(spec_values, columns=wavelength_cols)

# 添加样本 ID
sample_ids = [f"S{i+1:03d}" for i in range(n_samples_spec)]
df_spectra.insert(0, 'SampleID', sample_ids)

# 创建属性部分 DataFrame
target_names = ['Moisture', 'Starch', 'Oil', 'Protein']  # 请根据实际含义调整顺序！
df_targets = pd.DataFrame(prop_values, columns=target_names)

# 合并
df_combined = pd.concat([df_spectra, df_targets], axis=1)

# -------------------------------
# 5. 保存为 CSV
# -------------------------------
output_file = 'corn_mp5_regression_data.csv'
df_combined.to_csv(output_file, index=False)
print(f"\n✅ 成功保存整合数据到: {output_file}")
print(f"📊 数据形状: {df_combined.shape}")
print("📋 前几列预览:")
print(df_combined.iloc[:3, :6].to_string())  # 显示前3行，前6列