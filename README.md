# Model Generation and Evaluation for Hanzi Tasks

# Results Exhibit

​										   	**VEA**                                    **DDPM**                                   **GAN**  

<center class="half">
	<img src="picture/VAE_result.png" style="zoom:73%;" /><img src="picture/DDPM_results.png" style="zoom:75%;" /><img src="picture/GAN_results.png"  style="zoom:75%;" />
</center>




## 1. VAE  
### 1.1 Principle

$$
l_i (\theta, \phi) = -E_{z \sim q_{\theta}(z|x_i)} [\log p_{\phi}(x_i|z)] + KL(q_{\theta}(z|x_i) || p(z))
$$

<img src="picture/VAE_principle.png" alt="VAE_principle" style="zoom:48%;" />

### 1.2 Latent Space Visualization

- **Advantage of VAE**:  
  - Learns a continuous representation of input data in latent space.  
  - Enables smooth transitions between different characters.  

<center class="half">
	<img src="picture/VAE_Latent_space_visualization2.png" alt="VAE_Latent_space_visualization2" style="zoom:75%;" /><img src="picture/VAE_Latent_space_visualization.png" alt="VAE_Latent_space_visualization2" style="zoom:73%;" />
</center>
### 1.3 Training Loss

<center class="half">
	<img src="picture/VAE_training_loss.png" alt="VAE_Latent_space_visualization2" style="zoom:75%;" />
</center>

---



## 2. Diffusion Model  

### 2.1 Principle

<center class="half">
	<img src="picture/DDPM_principle1.png" alt="VAE_Latent_space_visualization2" style="zoom:70%;" />
    <img src="picture/DDPM_principle2.png" alt="VAE_Latent_space_visualization2" style="zoom:70%;" />
</center>



### 2.2 Diffusion Progress

![sampling_process](picture/sampling_process.gif)



### 2.3 Training Loss 

<center class="half">
	<img src="picture/DDPM_training_loss.png" alt="VAE_Latent_space_visualization2" style="zoom:75%;" />
</center>

---



## 3. GAN  

### 3.1 Principle

<center class="half">
	<img src="picture/GAN_principle.png" alt="VAE_Latent_space_visualization2" style="zoom:75%;" />
</center>

#### 3.2 Training Loss

<center class="half">
	<img src="picture/GAN_training_loss.png" alt="VAE_Latent_space_visualization2" style="zoom:75%;" />
</center>



## 4. Comparison & Evaluation  
### Evaluation Metrics  

#### **Accuracy**  
<img src="picture/accuracy.png" alt="accuracy" style="zoom:45%;" />



#### **Laplacian Variance as a Sharpness Indicator**  

- **Laplacian Operator**:  
  
  $$
  \nabla^2 I(x, y) = \frac{\partial^2 I(x, y)}{\partial x^2} + \frac{\partial^2 I(x, y)}{\partial y^2}
  $$
- **Laplacian Kernel**:

$$
\begin{bmatrix}0 & 1 & 0 \\ 1 & -4 & 1\\ 0 & 1 & 0 \end{bmatrix}
$$

- **Laplacian Variance**:  
  $$
  \sigma_L^2 = \frac{1}{N} \sum_{x=1}^{M} \sum_{y=1}^{N} (L(x,y) - \mu_L)^2
  $$
    

  ![sharpness](picture/sharpness.jpg)

### Training Process & Results  

| Model             | VAE    | GAN            | DDPM           |
| ----------------- | ------ | -------------- | -------------- |
| **Image Quality** | Blurry | Somewhat clear | Most realistic |
| **Convergence**   | Yes    | Fluctuating    | Yes            |

### Why is VAE Blurry?  
- Trade-off between reconstruction accuracy and latent space regularization.  
- Forces the model to prioritize smoothness and continuity in latent space.  



## References  

1. Ho, J., Jain, A., & Abbeel, P. (2020). **Denoising Diffusion Probabilistic Models**. [ArXiv](https://arxiv.org/abs/2006.11239)  
2. Diederik P. Kingma & Max Welling (2013). **Auto-Encoding Variational Bayes**. [ArXiv](https://arxiv.org/pdf/1312.6114)  
3. Ian J. Goodfellow (2014). **Generative Adversarial Nets**. [ArXiv](https://arxiv.org/abs/1406.2661)  
4. K. He, X. Zhang, S. Ren, & J. Sun. **Deep Residual Learning for Image Recognition**. CVPR 2016.  
5. D. Marziliano, F. Dufaux, S. Winkler, & T. Ebrahimi. **A fast method for image sharpness assessment based on the Laplacian operator**. ICIP 2002.  
