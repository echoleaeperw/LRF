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


## 🏗️ Results Display


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
