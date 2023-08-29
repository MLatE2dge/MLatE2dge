# 🔬 Machine Learning at the Extreme Edge - ML@E2dge

<p align="center">
  <img src="/img/mlate2dge.png" alt="MLatE2dge, This image was created with the assistance of DALL·E 2." width="300"/>
</p>

Today's trend is real-time and energy-efficient information extraction and processing at the edge using Artificial Intelligence. However, a recent trend exists to implement machine learning on devices located on the extreme edge, i.e. the border between the analog (physical) and digital world. These devices consist of one or more sensors and a resource-constrained embedded device, i.e. a device with limited memory, computing power, and power consumption. The challenge is the development of accurate, energy-efficient machine learning models for deployment on these resource-constrained devices. The project [🔗 Machine Learning @ the Extreme Edge](https://mlate2dge.github.io/) examines how to apply embedded machine learning to develop accurate, energy-efficient models for intelligent devices.


# ⚙️ Embedded Machine Learning Pipeline

<p align="center">
  <img src="/img/pipeline.png" alt="Pipeline"/>
</p>

**✏️ Note.** During the project's timeframe, retraining was performed using Edge Impulse Studio. In future implementations it is recommended to use the Profiling and Deploy [🔗 Edge Impulse Python SDK](https://docs.edgeimpulse.com/docs/tools/overview) (released April 4 2023 [Unveiling BYOM and the Edge Impulse Python SDK](https://edgeimpulse.com/blog/unveiling-the-new-edge-impulse-python-sdk)) combined with [🔗 Weights & Biases](https://docs.edgeimpulse.com/docs/integrations/weights-and-biases) AI developer platform. Some Python scripts can be found in [🔗 `./ei/profiling-deploy`](https://github.com/MLatE2dge/mlate2dge/tree/main/ei/profiling-deploy). These scripts can be used as a starting point for the integration into the embedded machine learning pipeline.

Link to the Python code: [🔗 ML@E2dge](https://github.com/MLatE2dge/mlate2dge)<br>

```
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

# 💻 Environment

The development was performed on a 64-bit Intel® Core™ i9-10900K CPU (20 cores), 3.70 GHz, 128 GB RAM, and an NVIDIA GeForce RTX3080 GPU type.<br>  

**Prerequisite**<br><br>
[🔗 Edge Impulse Studio](https://edgeimpulse.com/)<br>
[🔗 Weights & Biases platform](https://wandb.ai/)<br>

Create the environment using [🔗 conda](https://docs.conda.io/en/latest/miniconda.html). 

```
$ conda env create -f conda.yaml
```

**Recommended**<br><br>
[🔗 Visual Studio Code](https://code.visualstudio.com/)<br>
[🔗 Ubuntu 20.04.5 LTS (Focal Fossa)](https://cdimage.ubuntu.com/releases/focal/release/).

# References

## 🛠️ Tools
[🔗 Edge Impulse](https://edgeimpulse.com/)<br>
[🔗 Weights & Biases](https://wandb.ai/)<br>
[🔗 scikit-learn](https://scikit-learn.org/stable/)<br>
[🔗 TensorFlow](https://www.tensorflow.org/)<br>
[🔗 Keras](https://keras.io/)<br>
[🔗 pandas](https://pandas.pydata.org/)<br>
[🔗 pingouin](https://pingouin-stats.org/build/html/index.html)<br>
[🔗 matplotlib](https://matplotlib.org/)<br>
[🔗 bokeh](http://bokeh.org/)


## 📚 Books
[🔗 AI at the Edge](https://github.com/ai-at-the-edge)<br>
[🔗 Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)<br>
[🔗 Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition)<br>
[🔗 An Introduction to Statistical Learning](https://www.statlearning.com/)

## 🎓 Open Education
[🔗 Tiny Machine Learning Open Education Initiative (TinyMLedu)](http://tinyml.seas.harvard.edu/)

<br>

---
[🔗 Machine Learning @ the Extreme Edge](https://mlate2dge.github.io/) is a project supported by the Karel de Grote University of Applied Sciences and Arts through funding by the Flemish government specifically allocated to practice-based research at universities of applied sciences. 
<br> 📆 Project duration: 1 December 2021 until 31 August 2023 (14 person-month).

<div><p style="font-size: 11px"><a href="https://jrverbiest.github.io/">👨‍🔬 Principal Investigator: J.R. Verbiest</a></p></div>

Last page update: 29 August 2023