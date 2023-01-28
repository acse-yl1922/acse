# DL_module

This repository and README contains relevant information regarding the Deep Learning module, please read it all carefully. 

The machine learning framework we will use is [PyTorch](https://pytorch.org/).

# Table of contents
1. [Objectives](#objectives)
2. [Course contents](#coursecontents)
3. [Assessments](#assessments)
4. [Primer materials and bibliography](#primer)
5. [Lecture materials](#lectures)
6. [Google Colab](#colab)
7. [Teaching team](#team)
8. [Learning outcomes](#outcomes)


## Objectives <a name="objectives"></a>

The objective of this module is to equip you with a solid foundation to understand the basic principles of deep learning (DL).

Despite the humongous size of the deep-learning research landscape, and the impossibility to cover all the topics it encompasses, this course will provide you with all the fundamental concepts, building blocks, and mathematical methods that are used in the majority of modern complex applications, preparing you to develop your professional careers in this field.

## Course contents <a name="coursecontents"></a>

The course is structured in 6 blocks containing the following topics:

1. Introduction to PyTorch and recap of basic ML concepts (backprop, regularisation, etc).
2. Feed-forward Networks (FFNs)
3. Convolutional neural networks (CNNs)
4. Probability for DL
5. Generative models:
	- Variational autoencoders (VAEs)
	- Generative adversarial networks (GANs)
	- Diffusion models
6. Recurrent neural networks (RNNs) and long short-term memory networks (LSTMs)
7. Natural language processing (NLP)
8. Transformers
9. Deep reinforcement learning
10. State of the art in DL.

The above topics will be delivered using jupyter notebooks that contain both theory and practical implementations.

Most days will be structured as follows:

| time  | session |
|---------------|-----------------------------------------------------------------------|
| 9:00h-12:00h  | theory and implementation |
| 14:00h-17:00h | exercise session |


with a few exceptions:

On Wednesday afternoons we will not have any sessions.

The dates that will not follow the morning-lecture/afternoon-exercise structure will be:

| session  | activity |
|---------------|-----------------------------------------------------------------------|
| Friday 9 December afternoon | coursework 1 release and working time |
| Monday 12 December afternoon | coursework 1 working time |
| Thursday 15 December afternoon | review session of requested topics |
| Friday 16 December morning | Q&A and final preparations for coursework 2 |


## Assessments <a name="assessments"></a>

The module assessment is based on two courseworks.


| **COURSEWORK 1** | *4 days* |
| :------------------ | :---: |
| *Release date* | Friday 9 December 14:00h |
| *Due date (deadline)* | Monday 12 December 20:00h |


| **COURSEWORK 2** | *3 hours in class* |
| :------------------ | :---: |
| *Release date* | Friday 16 December 14:00h |
| *Due date* | Friday 16 December 17:00h |


**MARKS**: Your final mark will be the equally weighted average between coursework 1 and coursework 2.  




## Primer materials and bibliography <a name="primer"></a>

It is NOT mandatory to read (or view) any of the materials in this section.

### Introductory videos

To help prepare for the course, it is recommended to watch these four short (15-20 mins) videos which provide a good introduction to Machine Learning:

1. [But what is a Neural Network? | Deep learning, chapter 1](https://www.youtube.com/watch?v=aircAruvnKk)  
2. [Gradient descent, how neural networks learn | Deep learning, chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w&t=11s)   
3. [What is backpropagation really doing? | Deep learning, chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)  
4. [Backpropagation calculus | Deep learning, chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8)  

Probability introductory videos:

1. [Binomial distribution](https://www.youtube.com/watch?v=8idr1WZ1A7Q&list=RDCMUCYO_jab_esuFRV4b17AJtAw&index=2)
2. [Normal distribution](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data/more-on-normal-distributions/v/introduction-to-the-normal-distribution) 
3. [Bayes theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) 


### Bibliography
- [Pattern recognition and Deep Learning (Christopher M. Bishop)](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)
- [Deep Learning (Ian Goodfellow, Yoshua Bengio, Aaron Courville)](https://www.deeplearningbook.org/)



## Lecture materials <a name="lectures"></a>

The table below contains the course materials and their scheduled dates. Most notebooks will have two versions:

Lecture slides/notebooks will be released on the morning before the start of the lectures. Two notebooks will be provided:

- **Practical**/**Exercise notebook**: Template with some code provided and empty blocks to work on the implementation during the lecture (**Practical**) or during the afternoon exercises (**Exercise**).
- **Solutions notebook**: Practical/Exercise notebook completed with model solutions for the tasks done during the lecture or to be completed during the afternoon exercises.

We encourage you to work on the **Practical/Exercise notebook** and try to implement the tasks proposed during the lectures, but we acknowledge that some of you will prefer to focus on the delivered material and understand the code well before attempting to implement it yourselves.

*\[The numbers in the table (01:, 02:, etc), notebook names, Teams' meetings, and other materials, corresponds to the day of the course (from 01 to 15).\]*


| Session   |   Date <br> Time | Lecture/Exercises | Solutions     |
|-----------|:-------------------|-----------------------|----------------------------|
| **01: DL module intro** (morning)    |   Nov 28 <br> 09:00-12:00  |  [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/01_DL_module_intro/01_DL_module_intro_morning/01_Intro_Colab_and_FFNs_morning.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/01_DL_module_intro/01_DL_module_intro_morning/01_Intro_Colab_and_FFNs_morning.ipynb)                     |            | 
| **01: DL module intro** (afternoon)  |   Nov 28 <br> 14:00-17:00  |  [exercises](https://github.com/ese-msc-2022/DL_module/blob/main/01_DL_module_intro/01_DL_module_intro_afternoon/01_Intro_Colab_and_FFNs_afternoon_exercises.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/01_DL_module_intro/01_DL_module_intro_afternoon/01_Intro_Colab_and_FFNs_afternoon_exercises.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/01_DL_module_intro/01_DL_module_intro_afternoon/01_Intro_Colab_and_FFNs_afternoon_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/01_DL_module_intro/01_DL_module_intro_afternoon/01_Intro_Colab_and_FFNs_afternoon_solutions.ipynb)| 
| **02: PyTorch part I** (morning)     |   Nov 29 <br> 09:00-12:00  |   [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/02_PyTorch_part_I/02_PyTorch_part_I_morning/02_Pytorch_for_Deep_Learning_pt1_code_along.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/02_PyTorch_part_I/02_PyTorch_part_I_morning/02_Pytorch_for_Deep_Learning_pt1_code_along.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/02_PyTorch_part_I/02_PyTorch_part_I_morning/02_Pytorch_for_Deep_Learning_pt1.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/02_PyTorch_part_I/02_PyTorch_part_I_morning/02_Pytorch_for_Deep_Learning_pt1.ipynb)                         | 
| **02: PyTorch part I** (afternoon)   |   Nov 29 <br> 14:00-17:00  |   [exercise](https://github.com/ese-msc-2022/DL_module/blob/main/02_PyTorch_part_I/02_PyTorch_part_I_afternoon/02_Pytorch_for_Deep_Learning_aftenoon_exercise.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/02_PyTorch_part_I/02_PyTorch_part_I_afternoon/02_Pytorch_for_Deep_Learning_aftenoon_exercise.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/02_PyTorch_part_I/02_PyTorch_part_I_afternoon/02_Pytorch_for_Deep_Learning_aftenoon_exercise_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/02_PyTorch_part_I/02_PyTorch_part_I_afternoon/02_Pytorch_for_Deep_Learning_aftenoon_exercise_solutions.ipynb)        | 
| **03: PyTorch part II** (morning)    |   Nov 30 <br> 09:00-12:00  |    [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/03_PyTorch_part_II/03_PyTorch_part_II_morning/03_Pytorch_for_Deep_Learning_pt2_code_along.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/03_PyTorch_part_II/03_PyTorch_part_II_morning/03_Pytorch_for_Deep_Learning_pt2_code_along.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/03_PyTorch_part_II/03_PyTorch_part_II_morning/03_Pytorch_for_Deep_Learning_pt2.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/03_PyTorch_part_II/03_PyTorch_part_II_morning/03_Pytorch_for_Deep_Learning_pt2.ipynb)                                 | 
| **04: Recap ML concepts** (morning)  |   Dec  1 <br> 09:00-12:00  |    [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/04_Recap_ML_concepts/04_Recap_ML_concepts_morning/04_Recap_ML_concepts_morning_codealong.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/04_Recap_ML_concepts/04_Recap_ML_concepts_morning/04_Recap_ML_concepts_morning_codealong.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/04_Recap_ML_concepts/04_Recap_ML_concepts_morning/04_Recap_ML_concepts_morning_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/04_Recap_ML_concepts/04_Recap_ML_concepts_morning/04_Recap_ML_concepts_morning_solutions.ipynb)        | 
| **04: Recap ML concepts** (afternoon)|   Dec  1 <br> 14:00-17:00  |    [exercise](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/04_Recap_ML_concepts/04_Recap_ML_concepts_afternoon/04_Recap_ML_concepts_afternoon_exercises.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/04_Recap_ML_concepts/04_Recap_ML_concepts_afternoon/04_Recap_ML_concepts_afternoon_exercises.ipynb)                     | [solutions](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/04_Recap_ML_concepts/04_Recap_ML_concepts_afternoon/04_Recap_ML_concepts_afternoon_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/04_Recap_ML_concepts/04_Recap_ML_concepts_afternoon/04_Recap_ML_concepts_afternoon_solutions.ipynb)     | 
| **05: CNNs part I** (morning)        |   Dec  2 <br> 09:00-12:00  |         [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/05_CNNs_part_I/05_CNNs_part_I_morning/05_CNNs_part_I_morning_codealong.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/05_CNNs_part_I/05_CNNs_part_I_morning/05_CNNs_part_I_morning_codealong.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/05_CNNs_part_I/05_CNNs_part_I_morning/05_CNNs_part_I_morning_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/05_CNNs_part_I/05_CNNs_part_I_morning/05_CNNs_part_I_morning_solutions.ipynb)               | 
| **05: CNNs part I** (afternoon)      |   Dec  2 <br> 14:00-17:00  |           [exercises](https://github.com/ese-msc-2022/DL_module/blob/main/05_CNNs_part_I/05_CNNs_part_I_afternoon/05_CNNs_part_I_afternoon_exercises.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/05_CNNs_part_I/05_CNNs_part_I_afternoon/05_CNNs_part_I_afternoon_exercises.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/05_CNNs_part_I/05_CNNs_part_I_afternoon/05_CNNs_part_I_afternoon_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/05_CNNs_part_I/05_CNNs_part_I_afternoon/05_CNNs_part_I_afternoon_solutions.ipynb)                   | 
| **06: CNNs part II** (morning)       |   Dec  5 <br> 09:00-10:30  |      [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/06_CNNs_part_II/06_CNNs_part_II_morning/06_CNNs_part2_morning.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/06_CNNs_part_II/06_CNNs_part_II_morning/06_CNNs_part2_morning.ipynb)                     | <hr>           | 
| **06: Probability for DL** (morning) |   Dec  5 <br> 10:30-12:00  |    [slides](06_Probability/06_Probability_morning/06_Probability.pdf)                   |          <hr>          | 
| **06: CNNs part II** (afternoon)     |   Dec  5 <br> 14:00-17:00  |      [exercises](https://github.com/ese-msc-2022/DL_module/blob/main/06_CNNs_part_II/06_CNNs_part_II_afternoon/06_CNNs_part2_afternoon_exercises.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/06_CNNs_part_II/06_CNNs_part_II_afternoon/06_CNNs_part2_afternoon_exercises.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/06_CNNs_part_II/06_CNNs_part_II_afternoon/06_CNNs_part2_afternoon_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/06_CNNs_part_II/06_CNNs_part_II_afternoon/06_CNNs_part2_afternoon_solutions.ipynb)              | 
| **07: Autoencoders and VAEs** (morning)| Dec  6 <br> 09:00-12:00  |   [lecture slides](https://github.com/ese-msc-2022/DL_module/blob/main/07_Autoencoders_and_VAEs/07_Autoencoders_and_VAEs_morning/07_VAEs_theory_lecture.pdf)   <hr>  [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/07_Autoencoders_and_VAEs/07_Autoencoders_and_VAEs_morning/07_VAEs_morning_Exercise.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/07_Autoencoders_and_VAEs/07_Autoencoders_and_VAEs_morning/07_VAEs_morning_Exercise.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/07_Autoencoders_and_VAEs/07_Autoencoders_and_VAEs_morning/07_VAEs_morning_Solution.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/07_Autoencoders_and_VAEs/07_Autoencoders_and_VAEs_morning/07_VAEs_morning_Solution.ipynb)                        | 
| **07: Autoencoders and VAEs** (afternoon)|Dec 6 <br> 14:00-17:00  |       [exercise](https://github.com/ese-msc-2022/DL_module/blob/main/07_Autoencoders_and_VAEs/07_Autoencoders_and_VAEs_afternoon/07_VAEs_afternoon_Exercise.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/07_Autoencoders_and_VAEs/07_Autoencoders_and_VAEs_afternoon/07_VAEs_afternoon_Exercise.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/07_Autoencoders_and_VAEs/07_Autoencoders_and_VAEs_afternoon/07_VAEs_afternoon_Solution.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/07_Autoencoders_and_VAEs/07_Autoencoders_and_VAEs_afternoon/07_VAEs_afternoon_Solution.ipynb)              | 
| **08: GANs** (morning)               |   Dec  7 <br> 09:00-12:00  |       [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/08_GANs/08_GANs_morning/08_GANs_morning_codealong.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/08_GANs/08_GANs_morning/08_GANs_morning_codealong.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/08_GANs/08_GANs_morning/08_GANs_morning_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/08_GANs/08_GANs_morning/08_GANs_morning_solutions.ipynb)                                      | 
| **08: GANs** (additional material)               |   Dec  7 <br> 09:00-12:00  |       [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/08_GANs/08_GANs_additional_material/08_GANs_WGAN_GP_exercise.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/08_GANs/08_GANs_additional_material/08_GANs_WGAN_GP_exercise.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/08_GANs/08_GANs_additional_material/08_GANs_WGAN_GP_solution.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/08_GANs/08_GANs_additional_material/08_GANs_WGAN_GP_solution.ipynb)  | 
| **09: Diffusion models** (morning)   |   Dec  8 <br> 09:00-12:00  |    [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/09_Diffusion_models/09_Diffusion_models_morning/09_Diffusion_models_morning_codealong.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/09_Diffusion_models/09_Diffusion_models_morning/09_Diffusion_models_morning_codealong.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/09_Diffusion_models/09_Diffusion_models_morning/09_Diffusion_models_morning_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/09_Diffusion_models/09_Diffusion_models_morning/09_Diffusion_models_morning_solutions.ipynb)      | 
| **09: Diffusion models** (afternoon) |   Dec  8 <br> 14:00-17:00  |    [exercises](https://github.com/ese-msc-2022/DL_module/blob/main/09_Diffusion_models/09_Diffusion_models_afternoon/09_Diffusion_models_afternoon_exercises.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/09_Diffusion_models/09_Diffusion_models_afternoon/09_Diffusion_models_afternoon_exercises.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/09_Diffusion_models/09_Diffusion_models_afternoon/09_Diffusion_models_afternoon_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/09_Diffusion_models/09_Diffusion_models_afternoon/09_Diffusion_models_afternoon_solutions.ipynb)        | 
| **10: RNNs and LSTMs** (morning)     |   Dec  9 <br> 09:00-12:00  |    [slides](https://github.com/ese-msc-2022/DL_module/blob/main/10_RNNs_LSTMs/10_RNNs_LSTMs_morning/10_RNNs_LSTMs.pdf)  <hr>  [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/10_RNNs_LSTMs/10_RNNs_LSTMs_morning/10_RNNs_LSTMs_morning.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/10_RNNs_LSTMs/10_RNNs_LSTMs_morning/10_RNNs_LSTMs_morning.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/10_RNNs_LSTMs/10_RNNs_LSTMs_morning/10_RNNs_LSTMs_morning_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/10_RNNs_LSTMs/10_RNNs_LSTMs_morning/10_RNNs_LSTMs_morning_solutions.ipynb)             | 
| **10: RNNs and LSTMs** (afternoon)   |   Dec  9 <br> 14:00-17:00  |*coursework-1 time* |*coursework-1 time* | 
| **11: NLP** (morning)                |   Dec 12 <br> 09:00-12:00  |  [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/11_NLP/11_NLP_morning/11_NLP_morning_theory.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/11_NLP/11_NLP_morning/11_NLP_morning_theory.ipynb)                     |    [exercise](https://github.com/ese-msc-2022/DL_module/blob/main/11_NLP/11_NLP_morning/11_NLP_sentiment-analysis-with-glove_exercise.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/11_NLP/11_NLP_morning/11_NLP_sentiment-analysis-with-glove_exercise.ipynb)   <hr>  [solution](https://github.com/ese-msc-2022/DL_module/blob/main/11_NLP/11_NLP_morning/11_NLP_sentiment-analysis-with-glove_solution.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/11_NLP/11_NLP_morning/11_NLP_sentiment-analysis-with-glove_solution.ipynb) | 
| **11: NLP** (afternoon)              |   Dec 12 <br> 14:00-17:00  |*coursework-1 time* |*coursework-1 time* |
| **12: Transformers** (morning)       |   Dec 13 <br> 09:00-12:00  |    [lecture & exercise](https://github.com/ese-msc-2022/DL_module/blob/main/12_Transformers/12_Transformers_morning/12_Transformers_morning_codealong.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/12_Transformers/12_Transformers_morning/12_Transformers_morning_codealong.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/12_Transformers/12_Transformers_morning/12_Transformers_morning_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/12_Transformers/12_Transformers_morning/12_Transformers_morning_solutions.ipynb)  | 
| **12: Transformers** (afternoon)     |   Dec 13 <br> 14:00-17:00  |  *morning notebook*   | *morning notebook* | 
| **13: Deep RL** (morning)            |   Dec 14 <br> 09:00-12:00  |    [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/13_Deep_RL/13_Deep_RL_morning/13_Deep_RL_morning.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/13_Deep_RL/13_Deep_RL_morning/13_Deep_RL_morning.ipynb)                     | [solutions](https://github.com/ese-msc-2022/DL_module/blob/main/13_Deep_RL/13_Deep_RL_morning/13_Deep_RL_morning_solutions.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/13_Deep_RL/13_Deep_RL_morning/13_Deep_RL_morning_solutions.ipynb) | 
| **14: State of the art DL** (morning)|   Dec 15 <br> 09:00-12:00  |    [lecture](https://github.com/ese-msc-2022/DL_module/blob/main/14_State_of_the_art_DL/14_State_of_the_art_DL_morning/14_State_of_the_art_in_DL.ipynb) <br> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/ese-msc-2022/DL_module/blob/main/14_State_of_the_art_DL/14_State_of_the_art_DL_morning/14_State_of_the_art_in_DL.ipynb)     |          <hr>          | 
| **14: Review requests** (afternoon)  |   Dec 15 <br> 14:00-17:00  |                       |                    | 
| **15: Q&A and support CW2** (morning)|   Dec 16 <br> 09:00-12:00  | *session in class*    | *session in class* | 




###### The links in the table will become active as we progress during the course.

## Google Colab <a name="colab"></a>

All the coding will be done using Google Colab pro. It is also possible to use your own computer and run the jupyter notebooks locally, if you prefer, but limited support will be available to help you set up your local system.

There will be an introductory session on how to use Google Colab on day 01, and on day 03 we will go through the process of getting Colab Pro setup together in class.

#### **Do not buy any Colab Pro license as we will provide the method to do it in class**

#### **A new google account will be created on the first day of the class, which will be a dedicated account for the course, do NOT use your existing google accounts for this** 

## Teaching team <a name="team"></a>


- Lluis Guasch [(email)](mailto:lguasch@imperial.ac.uk) - ***module coordinator***
- Debbie Pelacani Cruz
- Carlos Cueto
- George Strong
- Oscar Bates
- Vinicius Santos Silva
- Amin Nadimy
- Siyi Li
- Thomas Davison



## Learning outcomes <a name="outcomes"></a>

#### Over the next three weeks, you will be able to go from here:


<img src="https://imgs.xkcd.com/comics/machine_learning.png" alt="drawing" width="300"/> <br>
[XKCD 1838](https://xkcd.com/1838/)

#### to understanding complex network architectures, how and why they work, and when to use them:

<img src="https://miro.medium.com/max/407/1*o0pS0LbXVw7i41vSISrw1A.png" alt="drawing" width="300"/> <br>
[Transformers](https://arxiv.org/pdf/1706.03762.pdf)








