# Modalities got latent: A novel approach to apply multi-head latent attention to multi-modal models.

Team IkAI presents to you a novel approach to solving multi-modal problems using what we call, *"Latent processing"*:
This approach was heavily inspired by the Multi-head Latent Attention paper as published by DeepSeek, back in 2024.
However, instead of applying latent attention technique only during the caching time, Latent processing compresses the Query, Key, and Value vectors into latent spaces at the very beginning before passing them to the transformer encoder blocks. As shown in the image below

![image](https://github.com/user-attachments/assets/79327f67-0100-4f47-92ae-e5ef1eed3b2e)

This approach not only makes training the thing easier, but also ensures faster evaluation.

Upon further improvement this model can be of much use in edge AI applications like, robotics. Upon arranging the proper hardware, and a more diverse dataset, this *approach* can certainly qualify to come in the big league to Vision-Language-Action models!

The Visual Question Answering model that we built using the approach is a mere demonstration of *Latent Processing's* capabilities...

Team IkAI members: 
Srijito Ghosh:- GitHub: https://www.github.com/Srijito354
Muskan Kumari:- GitHub: https://www.github.com/Muskan040399

In this project we tried building a Visual Question Answering (VQA) web-app using a CLIP model (built entirely from scratch), trained using the same *original to latent space compression technique*, as mentioned before. It was trained on the EasyVQA dataset (GH link: https://github.com/vzhou842/easy-VQA.git).

Libraries and Frameworks used:
Pytorch
Streamlit
