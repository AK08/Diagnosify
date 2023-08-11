<a name="readme-top"></a>
  [![Contributors][contributors-shield]][contributors-url]
  [![Forks][forks-shield]][forks-url]
  [![Stargazers][stars-shield]][stars-url]
  [![Issues][issues-shield]][issues-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/AK08/Disease-Classification.git">
    <img src="images/Medical research-cuate.png" alt="Logo" width="600px" >
  </a>

  <h3 align="center">Diagnosify - Disease Classification</h3>

  <p align="center">
    Welcome to our project!
    <br />
    <a href="https://github.com/AK08/Disease-Classification.git"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://drive.google.com/file/d/1FttMlM96K5njpDYt_v4MtVvyPX62REhj/view">View Video</a>
    ·
    <a href="https://disease-classification-iw6otvksjeyykhxy2kvmuw.streamlit.app/Pneumonia">View Demo</a>
    ·
    <a href="https://github.com/AK08/Disease-Classification.git">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
      <ul>
        <li><a href="#inspiration">Inspiration</a></li>
        <li><a href=#social-impact>Social Impact</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#intel-oneapi">Intel OneAPI</a>
      <ul>
        <li><a href="#use-of-onednn-in-our-project">Use of oneDNN and TensorFlow in our project</a></li>
      </ul>
    </li>
    <li><a href="#what-it-does">What it does</a></li>
    <li><a href="#how-we-built-it">How we built it</a></li>
    <li><a href="#what-we-learned">What we learned</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<div align="center">
  <img src="images/SmartGarbage.gif" type="gif" alt="png" width="750">
</div>
Diagnosify is an innovative disease classification project designed to leverage the power of machine learning and Intel oneAPIs for accurate and efficient disease prediction. In the modern healthcare landscape, early and accurate diagnosis plays a pivotal role in ensuring timely medical interventions and improved patient outcomes. Diagnosify addresses this need by offering a platform that assists medical professionals in diagnosing various diseases using advanced machine learning techniques. The project focuses on the classification of diseases such as Brain Tumor, Alzheimer's Disease, Chest Diseases, Skin Diseases, and Lower Back Pain. This repository contains the code and resources used to train and deploy the disease classification models.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Inspiration <img src="images/inspo.png" alt="png" width="30">
The inspiration behind the creation of Diagnosify stems from the pressing need for more accurate, efficient, and accessible methods of disease diagnosis in the modern healthcare landscape. The convergence of medical expertise and technological advancements has the potential to revolutionize the way diseases are detected and treated. Several key factors have motivated the inception of Diagnosify:

1. <b>Early Detection and Intervention: </b> Timely detection of diseases is often a critical factor in determining patient outcomes. Many medical conditions, such as brain tumors and neurodegenerative disorders like Alzheimer's disease, can exhibit subtle symptoms in their early stages. Diagnosify seeks to provide healthcare professionals with tools that enable them to identify these diseases at an early phase, leading to more effective treatment strategies and improved patient quality of life.

2. <b>Limited Resources and Expertise: </b>In various regions around the world, access to specialized medical expertise is limited, especially in rural or underserved areas. Diagnosify's technology-driven approach aims to bridge this gap by providing medical professionals with supplementary diagnostic tools. These tools can aid in making accurate assessments and decisions, even in areas with fewer medical resources.

3. <b>Technological Advancements in Medical Imaging: </b> The rapid advancements in medical imaging technologies, such as MRI, CT scans, and dermatological imaging, have led to an abundance of data that can potentially be harnessed for disease diagnosis. Diagnosify leverages these technologies to analyze intricate medical images and data, extracting valuable insights that can aid in disease classification.

### Social Impact <img src="images/impact.png" alt="png" width="30">
The "Diagnosify - Disease Classification" project has the potential to create significant social impact in several ways:

1. Early Detection and Treatment: By accurately predicting diseases like brain tumors, Alzheimer's, chest diseases, skin diseases, and lower back pain, the project can enable early detection and intervention. Early diagnosis often leads to more effective treatment options and improved patient outcomes.
2. Healthcare Access: Diagnosify can extend medical expertise to underserved and remote areas where access to specialized healthcare may be limited. This democratization of medical diagnosis can ensure that individuals in various geographical locations receive timely and accurate predictions.
3. Reduced Misdiagnosis: Machine learning algorithms used in the project can help reduce instances of misdiagnosis by analyzing intricate patterns that might be challenging for human experts to identify. This can prevent unnecessary treatments and procedures while increasing the accuracy of diagnoses.

4. Collaborative Healthcare: The project promotes collaboration between medical professionals and technology. This synergy can lead to a more comprehensive understanding of diseases and their characteristics, fostering a collaborative approach to healthcare.

### Built With <img src="images/built.png" alt="png" width="30">

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![oneapi][oneapi]][oneapi-url]
  * [![onednn][onednn]][onednn-url]
* [![python][python]][python-url]
* [![jupyter][jupyter]][jupyter-url]
* [![tensorflow][tensorflow]][tensorflow-url]
* [![streamlit][streamlit]][streamlit-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Intel one api -->
## Intel oneAPI <img src="images/oneapi2.png" alt="png" width="30">
Intel OneAPI is a comprehensive development platform for building high-performance, cross-architecture applications. It provides a unified programming model, tools, and libraries that allow developers to optimize their applications for Intel CPUs, GPUs, FPGAs, and other hardware. Intel OneAPI includes support for popular programming languages like C++, Python, and Fortran, as well as frameworks for deep learning, high-performance computing, and data analytics. With Intel OneAPI, developers can build applications that can run on a variety of hardware platforms, from edge devices to data centers, and take advantage of the performance benefits of Intel architectures.

### Use of oneDNN and TensorFlow in our project
<img src="images/onednn.png" alt="png" width="700">
OneDNN provides highly optimized routines for various deep learning operations, including convolution, pooling, normalization, and activation functions. By using oneDNN, you can expect faster execution times and better performance on modern CPUs, especially those with Intel processors.

In this project <code>os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'</code> line sets an environment variable called <code>TF_ENABLE_ONEDNN_OPTS to '1'</code>. This enables the use of Intel's OneAPI Deep Neural Network Library (OneDNN) optimizations for TensorFlow on the system where this code is being run. OneDNN is a high-performance library for deep learning that is designed to optimize the performance of deep neural network computations on a variety of hardware platforms. By enabling OneDNN optimizations, this code may run faster on certain hardware architectures that are compatible with OneDNN. <strong>In this project, the Conv2D and Dense layers will be automatically optimized using oneDNN, which should result in faster training and inference times on compatible hardware.</strong>

The <code>tensorflow.keras</code> module is used to create a convolutional neural network (CNN) model for image classification. The model architecture consists of three convolutional blocks, each followed by a max pooling layer, and three fully connected layers with dropout for regularization.

Finally, the <code>model.compile</code> method is called to configure the optimizer, loss function, and evaluation metric for the model. The optimizer used is Adam, and the loss function used is sparse categorical cross-entropy. The model is also evaluated using the accuracy metric.

<!-- What it does -->
## What it does <img src="images/does.png" alt="png" width="30">
Diagnosify employs sophisticated machine learning algorithms to analyze medical data and images. This enables it to deliver accurate predictions about disease presence, assisting medical experts in making informed decisions. The project develops specialized machine learning models for each disease category, finely tuned to recognize distinct disease characteristics. By integrating Intel oneAPIs, Diagnosify optimizes its performance, ensuring efficient use of hardware resources for quicker and more reliable predictions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## How we built it <img src="images/built.png" alt="png" width="30">
These are the steps involved in making this project: 
* Importing Libraries
* Data Importing
* Data Exploration
* Data Configuration
* Preparing the Data
  * Creating a Generator for Training Set
  * Creating a Generator for Testing Set
* Writing the labels into a text file 'Labels.txt'
* Model Creation
* Model Compilation
* Training the Model (batch_size = 32, epochs = 10)
* Testing Predictions
* Saving model as 'modelnew.h5'
* Deploying the Model as a Web Application using Streamlit

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## What we learned <img src="images/learn.png" alt="png" width="30">
✅Building Diagnosify using oneDNN and Intel oneAPIs has been a transformative journey, providing us with a deep understanding of cutting-edge technologies and their practical applications in the field of disease classification. Here's a summary of key learnings from this experience:

✅ Hardware Optimization Expertise: Working with oneDNN and Intel oneAPIs exposed us to advanced techniques for optimizing machine learning models on diverse hardware architectures. We gained insights into harnessing the full potential of CPUs, GPUs, and other accelerators, enhancing our ability to design efficient and high-performance solutions.

✅Performance-Centric Mindset: Integrating oneDNN taught us to think critically about performance bottlenecks and resource utilization. We learned to fine-tune our models, optimize memory usage, and leverage hardware-specific features to achieve optimal inference speeds.

✅Hardware-Agnostic Deployment: The ability to deploy our models seamlessly on various hardware architectures showcased the power of hardware-agnostic solutions. We gained confidence in creating versatile applications that can adapt to different computing environments.

✅Model Evaluation:  Working with oneDNN and Intel oneAPIs encouraged us to iterate on model architectures and hyperparameters. We gained proficiency in fine-tuning models for optimal accuracy and performance, resulting in refined disease prediction capabilities.

✅Educational Impact: The project's use of advanced technologies like oneDNN and Intel oneAPIs presented opportunities for educational outreach. We learned to convey complex technical concepts to wider audiences, promoting awareness of AI's potential in healthcare.

✅Innovation at the Intersection: Diagnosify's creation at the intersection of medicine and technology highlighted the potential for innovative solutions that bridge disciplines. We gained insights into the challenges and rewards of interdisciplinary projects.


In conclusion, our journey of building Diagnosify using oneDNN and Intel oneAPIs has been a transformative experience that has enriched our understanding of cutting-edge technologies, healthcare applications, and the profound impact of responsible AI integration. This project has yielded a diverse array of insights, fostering growth in technical expertise, ethical considerations, collaboration, and real-world problem-solving. Through this endeavor, we have not only created a disease classification platform but also embarked on a significant learning journey with enduring implications.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/kottaram007/Smart-Garbage-Segregation.svg?style=for-the-badge
[contributors-url]: https://github.com/kottaram007/Smart-Garbage-Segregation/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/kottaram007/Smart-Garbage-Segregation.svg?style=for-the-badge
[forks-url]: https://github.com/kottaram007/Smart-Garbage-Segregation/network/members
[stars-shield]: https://img.shields.io/github/stars/kottaram007/Smart-Garbage-Segregation.svg?style=for-the-badge
[stars-url]: https://github.com/kottaram007/Smart-Garbage-Segregation/stargazers
[issues-shield]: https://img.shields.io/github/issues/kottaram007/Smart-Garbage-Segregation.svg?style=for-the-badge
[issues-url]: https://github.com/kottaram007/Smart-Garbage-Segregation/issues

[product-screenshot]: images/screenshot.png



[python]: https://img.shields.io/badge/Python-3470a3?&logoColor=white
[python-url]: https://www.python.org/
[jupyter]: https://img.shields.io/badge/Jupyter%20Notebook-da5b0b?&logoColor=white
[jupyter-url]: https://jupyter.org/
[tensorflow]: https://img.shields.io/badge/TensorFlow-f0b93a?&logoColor=white
[tensorflow-url]: https://www.tensorflow.org/
[streamlit]: https://img.shields.io/badge/Streamlit-f24747?&logoColor=white
[streamlit-url]: https://streamlit.io/
[oneapi]: https://img.shields.io/badge/Intel%20oneAPI-20232A?&logoColor=61DAFB
[oneapi-url]: https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-0/intel-oneapi-data-analytics-library-onedal.html
[onednn]: https://img.shields.io/badge/oneDNN-20232A?&logoColor=61DAFB
[onednn-url]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html

