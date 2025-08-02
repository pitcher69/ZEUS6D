# ZEUS6D: Zero‑Shot Estimation of Unseen Shapes for 6‑DoF Pose Recovery

**ZEUS6D** is a *training‑free* deep‑learning pipeline that estimates an object’s full 6‑DoF pose from a **single RGB image**, using only its CAD model—no object‑specific fine‑tuning required.

---

## 🔍 What is 6‑DoF Pose Estimation?

6‑DoF pose describes an object’s

* **3 DoF rotation** – roll, pitch, yaw
* **3 DoF translation** – \$x\$, \$y\$, \$z\$

Together they form a rigid‑body transform:

$$
\mathbf T \,=\,
\begin{bmatrix}
\mathbf R & \mathbf t \\
\mathbf 0^{\!\top} & 1
\end{bmatrix}
$$

with \$\mathbf R\in SO(3)\$ and \$\mathbf t = (t\_x, t\_y, t\_z)\$.  In expanded form:

$$
\begin{bmatrix}
 r_{11} & r_{12} & r_{13} & t_x\\
 r_{21} & r_{22} & r_{23} & t_y\\
 r_{31} & r_{32} & r_{33} & t_z\\
 0 & 0 & 0 & 1
\end{bmatrix}
$$

---

## 🛠 Why It Matters

Accurate 6‑DoF poses unlock

* Robotics: manipulation & grasp planning
* AR / VR: stable virtual anchors
* 3‑D scene understanding & reconstruction
* Industrial metrology & automated inspection

**ZEUS6D** delivers these poses *zero‑shot*: swap in a new CAD mesh and infer immediately—no retraining cycle.

---

## 📽️ Examples

**AR application**

![AR demo](examples/317560123-80e96855-a73c-4bee-bcef-7cba92df55ca.gif)

**Pipeline demo**

![Pipeline comparison](examples/my_comparison_animation.gif)

---

> *For a full technical deep‑dive—pipeline stages, datasets, failure modes—see the documentation site in `docs/` or visit the published GitHub Pages site.*
