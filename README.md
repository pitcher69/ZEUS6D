# ZEUS6D: Zero-shot Estimation of Unseen Shapes for 6DoF Pose Recovery

**ZEUS6D** is a training-free deep learning pipeline for estimating the **6D pose** of known objects from a single RGB image using their CAD models. The framework is designed to work without object-specific training or fine-tuning, enabling fast and scalable deployment in real-world scenarios.

## 🔍 What is 6D Pose Estimation?

**6D pose estimation** refers to determining the full 3D position and orientation of an object with respect to the camera. Specifically, it estimates:
- **3 degrees of freedom (DoF)** for rotation (roll, pitch, yaw)
- **3 DoF** for translation (x, y, z coordinates)

Together, these form a transformation:

$$
\begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
$$

More precisely, this is a $4 \times 4$ homogeneous transformation matrix of the form:

$$
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

Where:
- $R$ is a $3 \times 3$ rotation matrix
- $t = (t_x, t_y, t_z)$ is the 3D translation vector

### 🛠 Why It Matters

Accurate 6D pose estimation is critical for:
- Robotic manipulation and grasping
- Augmented and virtual reality
- Scene understanding and 3D reconstruction
- Industrial automation and inspection

Unlike conventional methods that rely on object-specific training data, **ZEUS6D** enables **zero-shot inference** for novel objects using only their CAD models — no retraining required.

## 📽️ Examples

AR Applications:  
https://github.com/NVlabs/FoundationPose/assets/23078192/80e96855-a73c-4bee-bcef-7cba92df55ca

