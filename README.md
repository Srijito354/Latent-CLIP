# Modalities got latent: A novel approach to latent space for more efficient usage of resources in multi-modal models.

We present to you a novel approach to solving multi-modal problems using what we call, *"Latent processing"*:
This approach was heavily inspired by the Multi-head Latent Attention paper as published by DeepSeek, back in 2024.
However, instead of applying latent attention technique only during the caching time, Latent processing compresses the Query, Key, and Value vectors into latent spaces at the very beginning before passing them to the transformer encoder blocks. As shown in the image below

![image](https://github.com/user-attachments/assets/79327f67-0100-4f47-92ae-e5ef1eed3b2e)
![image](https://github.com/user-attachments/assets/07816b8c-3bad-4a97-970f-4520bbe57d7e)


This approach not only makes training the thing easier, but also ensures faster inference on lower end hardware in a world where higher-end GPUs are getting expensive and difficult to procrure with each coming day...

Upon further improvement this model will be of much use in edge AI applications like, robotics. With proper robotics hardware, mated with a more diverse real-world dataset for scene reasoning (like GQA), this *approach* can certainly qualify to come in the big league of Vision-Language-Action models!

The Visual Question Answering model that we built using the approach is a mere demonstration of *Latent Processing's* capabilities...

Team IkAI members: 
Srijito Ghosh:- GitHub: ![link](https://www.github.com/Srijito354)
Muskan Kumari:- GitHub: ![link](https://www.github.com/Muskan040399)

In this project we tried building a Visual Question Answering (VQA) web-app using a CLIP model (built entirely from scratch), trained using the same *original to latent space compression technique*, as mentioned before. It was trained on the EasyVQA dataset (GH link: https://github.com/vzhou842/easy-VQA.git).

Libraries and Frameworks used:
Pytorch
Streamlit

To use the model, run "streamlit run app.py" in the terminal, after installing the necessary libraries and frameworks as mentioned above.

Note: The model was trained in WSL (Windows Subsystem for Linux). However, please make sure use the repo in Windows to make proper use of it. Thank you!
