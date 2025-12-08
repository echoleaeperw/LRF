# Learning from Risk: LLM-Guided Generation of Safety-Critical Scenarios with Prior Knowledge

![LFR_Framework](Fig_1.png)

     
## Installation

1. Create conda environment

   ```bash
    conda create -n lrf python=3.8
    ```
3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Initial scene generation

nuscenes：
 ```bash
    python src/train_traffic.py --config ./configs/train_traffic.cfg
 ```

---------------

## Generation of scenarios with different risk levels

Run the following command, and you can replace llm_model with your own model.

     ```bash
    python src/adv_scenario_gen.py \
         --config configs/adv_gen_rule_based.cfg \
         --ckpt model_ckpt/traffic_model.pth \
         --use_llm \
         --llm_model deepseek-reason
     ```


## 4. TODO List
<div align="center">

## 🏗️ Qualitative Results

<!-- 
  布局策略：
  - 只展示 4 组最精华的结果
  - 保持 width="50%" 的大尺寸，确保细节清晰可见
  - 移除了冗余的表头，用简洁的图注说明
-->

<table>
  <tr>
    <th width="50%" style="text-align:center"><strong>Method A (Ours)</strong></th>
    <th width="50%" style="text-align:center"><strong>Method B (Baseline)</strong></th>
  </tr>
  
  <!-- Group 1 -->
  <tr>
    <td><video src="https://github.com/user-attachments/assets/0179f497-9c1b-43e5-aa0e-3727f1973d1d"></td>
    <td><video src="https://github.com/user-attachments/assets/6a02e551-d16e-4f79-a111-a921f42e979c"></td>
  </tr>
  
  <!-- Group 2 -->
  <tr>
    <td><video src="https://github.com/user-attachments/assets/892bfb84-2771-4298-86ff-60a5a4c8ad5b"></td>
    <td><video src="https://github.com/user-attachments/assets/5718b287-2205-454c-bb7d-a13a6ce5bc10"></td>
  </tr>

  <!-- Group 3 -->
  <tr>
    <td><video src="https://github.com/user-attachments/assets/156efd88-a3a4-46c0-bc40-5adc4a2345c7"></td>
    <td><video src="https://github.com/user-attachments/assets/e01d31fb-9e25-42b9-a858-9108a6d012af"></td>
  </tr>

  <!-- Group 4 -->
  <tr>
    <td><video src="https://github.com/user-attachments/assets/089d4332-5972-4cd3-9b47-e482e70bdfe6"></td>
    <td><video src="https://github.com/user-attachments/assets/054f7016-7199-40ba-b2d9-109aecf04f24"></td>
  </tr>

</table>

</div>
## 📄 引用

如果使用本项目，请引用：

```bibtex
@inproceedings{,
  title={Learning from Risk: LLM-Guided Generation of Safety-Critical Scenarios with Prior Knowledge},
  author={...},
  booktitle={CVPR},
  year={2022}
}
```


## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件



**注意**：本项目用于学术研究目的。在自动驾驶系统中部署前，请进行充分的安全测试。

最后更新：2025年11月10日
