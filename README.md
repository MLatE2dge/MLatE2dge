# 🔬 Machine Learning at the Extreme Edge - ML@E2dge

<p align="center">
  <img src="/img/mlate2dge.png" alt="MLatE2dge, This image was created with the assistance of DALL·E 2." width="300"/>
</p>

Today's challenge is real-time and energy-efficient information extraction and processing at the edge using Artificial Intelligence. However, a recent trend exists to implement machine learning on devices located on the extreme edge, i.e. the border between the analog (physical) and digital world. These devices consist of one or more sensors and a resource-constrained embedded device, i.e. a device with limited memory, computing power, and power consumption. Today's challenge is the development of accurate, energy-efficient machine learning models for deployment on these resource-constrained devices. The project [🔗 Machine Learning @ the Extreme Edge](https://mlate2dge.github.io/) examines how to apply embedded machine learning to develop accurate, energy-efficient models for intelligent devices.

Project website: [🔗 Machine Learning @ the Extreme Edge](https://mlate2dge.github.io/)<br>
The code: [🔗 ML@E2dge](https://github.com/MLatE2dge/mlate2dge)<br>

```
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


# ⚙️ Embedded Machine Learning Pipeline

<p align="center">
  <img src="/img/pipeline.png" alt="Pipeline"/>
</p>

**✏️ Note.** During the project's timeframe, retraining was performed using Edge Impulse Studio. In future implementations it is recommended to use the Profiling and Deploy from [🔗 Edge Impulse Python SDK](https://docs.edgeimpulse.com/docs/tools/overview) combined with [🔗 Weights & Biases](https://docs.edgeimpulse.com/docs/integrations/weights-and-biases), some Python scripts can be found in [🔗 `./ei/profiling-deploy`](https://github.com/MLatE2dge/mlate2dge/tree/main/ei/profiling-deploy). These scripts can be used as a starting point for the integration into the pipeline.


# 💻 Environment Setup

**Prerequisite:**<br><br>
[🔗 Edge Impulse Studio](https://edgeimpulse.com/)<br>
[🔗 Weights & Biases platform](https://wandb.ai/)<br>

**Recommended:**<br><br>
[🔗 Visual Studio Code](https://code.visualstudio.com/)<br>

<br>The developement is done using the Ubuntu OS (Ubuntu 20.04.5 LTS). Create the environment using [🔗 conda](https://docs.conda.io/en/latest/miniconda.html): 

```
$ conda env create -f conda.yaml
```

# References

## 🛠️ Tools
[🔗 Edge Impulse](https://edgeimpulse.com/)<br>
[🔗 Weights & Biases](https://wandb.ai/)<br>
[🔗 TensorFlow](https://www.tensorflow.org/)<br>
[🔗 Keras](https://keras.io/)<br>
[🔗 pandas](https://pandas.pydata.org/)<br>
[🔗 pingouin](https://pingouin-stats.org/build/html/index.html)<br>
[🔗 scikit-learn](https://scikit-learn.org/stable/)<br>

## 📚 Books
[🔗 AI at the Edge](https://www.oreilly.com/library/view/ai-at-the/9781098120191/)<br>
[🔗 Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)<br>
[🔗 Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition)


---
🔬 [🔗 Machine Learning @ the Extreme Edge](https://mlate2dge.github.io/) is a project supported by the Karel de Grote University of Applied Sciences and Arts through funding by the Flemish government specifically allocated to practice-based research at universities of applied sciences. ⌛ Project duration: 1 December 2021 until 31 August 2023 (14 person-month).