![neuroCraft](raw_data/NeuroCraft_creamy.png)
**LeWagon** - Data Science Project.
<br>
<br>
Simplifying the lifes of neurodivergent learners.

## Overview
NeuroCraft aims to create a model to classify texts regarding their easiness for dyslexia readers. After testing different models, we have chosen a [RoBERTa model](https://huggingface.co/docs/transformers/model_doc/roberta). The model is trained on the [CLEAR (CommonLit Ease of Readability) Corpus](https://docs.google.com/spreadsheets/d/1sfsZhhP2umXXtmEP_NRErxLuwgN98TyH7LWOq3j07O0/edit?ref=commonlit.org), an open dataset of 5,000 excerpts for students from grades 3 to 12 in English Language Arts classrooms.

For the simplification strategy, we have used the GPT-4-Turbo model, employing a Zero-Shot & Few-Shot Prompting approach to assess text difficulty levels. To provide a comprehensive solution, we offer texts at various difficulty levels, using them as benchmarks for comparison. To ensure dyslexic friendliness, we have established clear guidelines for text simplification, with a focus on creating an accessible Large Language Model (LLM) framework.

We are committed to continuous improvement and have identified areas for enhancement:
- Improve Classification Models: Experiment with larger embeddings to enhance the model's understanding and predictive capabilities.
- Develop plugins for web browsers to seamlessly integrate NeuroCraft into users' online experiences.

Curious to learn more? Explore our platform [here](https://neurocraft.streamlit.app/)!

## About
NeuroCraft was co-created by [Andrea Calcagni](https://github.com/AndreaCalcagni), [Pei-Yu Chen](https://github.com/renee1j), [Patricia Moreno Gaona](https://github.com/patmg-coder) and [Christine Sigrist](https://github.com/ChristineSi) as the final project for the Data Science bootcamp at [Le Wagon](https://www.lewagon.com/) (batch 1451) on December 2023. The code is written in Python, hosted and published with [Streamlit](https://streamlit.io/).
