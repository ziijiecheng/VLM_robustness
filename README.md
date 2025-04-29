# Understanding the robustness of vision-language models to medical image artefacts
The project consists of two core components. The first involves constructing benchmarks with various image artefacts. The second focuses on applying VLMs in disease detection tasks.

The code for the first component is available in `imaging_artefacts.ipynb`. It allows you to apply intensity artefacts—including random noise, bias fields, and motion—to original, unaltered images. We implemented it using [torchio](https://github.com/TorchIO-project/torchio) as a reference. You can also apply spatial artefacts such as random cropping and rotation. The benchmark used in this study is avaliable at: https://drive.google.com/drive/folders/1M7EldoSvxEMZ2jA9wJs52H1-4G2zTU8C?usp=sharing.

The code for the second component is available in `VLMs_evaluation.ipynb`. It utilizes vision-language models (VLMs), including [GPT-4o](https://platform.openai.com/docs/quickstart), [Claude 3.5 Sonnet](https://github.com/anthropics/anthropic-cookbook), [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), and [Llama 3.2 11B](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct). You can experiment with various prompts provided in `prompt.md`.

Please contact: rmapzch@ucl.ac.uk if you have any questions.
![VLM](https://github.com/user-attachments/assets/65e2d944-31eb-4b3b-aa22-9874fac5205e)

## 🔑 Key Features

- **Integrated Framework**: 1) Benchamrk construction 2) Application of VLMs in disease detection 3) Metrics for robustness evaluation

- **Customizable**: Easily extended to other medical fields

## 🧪 Evaluate VLMs' Robustness on Other Medical Fields

  1.Build original unaltered images dataset in structure:
 original_unaltered_images/
├── normal/
└── diseased/
	2.Introducing image artefacts following `imaging_artefacts.ipynb` to your original unaltered images to construct benchmarks in strucuture:
 dataset/
├── weak_artifacts/
│   ├── bias_field/
│   │   ├── normal/
│   │   └── diseased/
│   ├── random_noise/
│   │   ├── normal/
│   │   └── diseased/
│   ├── motion/
│   │   ├── normal/
│   │   └── diseased/
│   ├── rotation/
│   │   ├── normal/
│   │   └── diseased/
│   └── cropping/
│       ├── normal/
│       └── diseased/
└── strong_artifacts/
    ├── bias_field/
    │   ├── normal/
    │   └── diseased/
    ├── random_noise/
    │   ├── normal/
    │   └── diseased/
    ├── motion/
    │   ├── normal/
    │   └── diseased/
    ├── rotation/
    │   ├── normal/
    │   └── diseased/
    └── cropping/
        ├── normal/
        └── diseased/
  
	3. Evaluated VLMs' performance at each artefacts.


