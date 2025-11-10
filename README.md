# Learning from Risk: LLM-Guided Generation of Safety-Critical Scenarios with Prior Knowledge



     

```

## 🚀 快速开始
### 1. 环境设置

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/LRF.git
cd LRF

# 创建conda环境
conda create -n lrf python=3.8
conda activate lrf

# 安装依赖
pip install -r requirements.txt
```

### 2. 初始场景生成


### 3. 对抗场景生成

结合梯度优化和LLM引导的场景生成：

```bash
python src/adv_scenario_gen.py \
    --config configs/adv_gen_rule_based.cfg \
    --ckpt model_ckpt/traffic_model.pth \
    --use_llm \
    --llm_model deepseek-chat
```
### 4.TODO List

## 🏗️ 项目结构


LRF/
├── configs/              
│   ├── llm_config.json          
│   ├── llm_weights_config.yaml   
│   ├── adv_gen_*.cfg             
│   └── eval_planner.cfg          
├── src/                  
│   ├── models/          
│   ├── losses/          
│   ├── datasets/       
│   ├── planners/        
│   ├── llm/             
│   └── utils/           
├── longterm/            
│   ├── agents/          
│   │   ├── analysis.py     
│   │   ├── driver.py      
│   │   ├── flow.py         
│   │   └── reflection.py   
│   ├── core/           
│   │   ├── llm_factory.py     
│   │   ├── json_parser.py      
│   │   └── content_processor.py # 内容处理
│   └── knowledge/      
│       ├── behavior_corpus.json    
│       └── scenario_physics_knowledge_base.json
├── data/               
│   ├── nuscenes/       
│   ├── clustering/     
│   └── scenarios/      
├── model_ckpt/         
├── outputs/            
├── logs/               
├── evaluate_traffic_model_cvae.py  
├── run_adversarial_evaluation.py   
├── visualize_cvae_evaluation.py    
├── requirements.txt    
├── LICENSE             
└── README.md  

## 📄 引用

如果使用本项目，请引用：

```bibtex
@inproceedings{ding2022strive,
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
