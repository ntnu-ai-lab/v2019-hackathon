# AI Hackathon

<img src="https://i.imgur.com/w11MC1p.png" height="50" 
style="display:inline-block;"></img>
<img src="https://i.imgur.com/rpFBAmZ.png" height="50" 
style="transform: translateY(-30%); margin: 0px 20px;"></img>
<img src="https://i.imgur.com/oSZhtAW.jpg" height="50" 
style="display:inline-block; position: relative; transform: translateY(-20%);"></img>

#### With BRAIN NTNU, Telenor and Norwegian Open AI Lab

**Time:** 1st of March through 3rd
**Place:** R2, Gløshaugen NTNU, Trondheim

**Timeline:**

*Friday:*

* 15:15 - Presentation of assignment
* 16:00 - Competition begins, R2 opens
* 19:00 - Dinner

*Saturday:*

* 12:00 - Competition day 2 begins, R2 opens
* 14:00 - Lunch
* 18:00 - Dinner
* 23:59 - Deadline for delivery
 
*Sunday:*

* 13:00 - Awards ceremony

### The assignment

Monitoring and predicting the air quality level (NO<sub>x</sub>, PM<sub>10</sub> and PM<sub>2.5</sub>)​  in the city and making it available to the public would increase the awareness of people living in the city about how good their environment is. From that, it would engage them to be part of solutions to reduce the pollution in the city. Monitoring and forecasting air quality would also benefit decision makers in the city to propose proper and systematic solutions to reduce the emission, such as reducing cars in different time of a day or week. The main objective of this hackathon is to build an application for air quality monitoring and prediction in Trondheim area for those purposes. With the hackathon, students will have ability to work with NB-IoT sensors developed by Telenor and work with data (NO<sub>x</sub>, PM<sub>10</sub> and PM<sub>2.5</sub>)​ collected and available at AI-lab. The students can also utilize the IoT analytics platform setup by Telenor to AI-lab to facilitate the data processing and data visualization. 


Telenor is the second biggest IoT provider in Europe with more than 10 millions IoT devices for more than 100 customers. With ambition to empower society, we are currently helping to build smart municipalities by using IoT technologies. Working on this project, students will have an opportunity to understand Telenor's business, learning more about IoT, Big Data and AI/Machine Learning. 



> **Relevant articles:**
> 
> * <a href="https://norden.diva-portal.org/smash/get/diva2:1069152/FULLTEXT02.pdf">Road dust and PM<sub>10</sub> in the Nordic countries</a>
> * <a href="https://brage.bibsys.no/xmlui/handle/11250/235839">Pavement wear and airborne dust pollution in Norway</a>
> * <a href="https://chemicalwatch.com/66144/norway-investigating-solutions-for-tackling-road-dust-microplastics">Norway investigating solutions for tackling road dust microplastics
</a>


### Preliminaries

##### Sharing computational power - MANDATORY FOR ALL TEAMS

Since we are all sharing NTNUs HPC we have to distribute the processing power amongst all teams. You do this by setting the fraction of GPU memory to be allocated when construct a `tf.Session` (if using Tensorflow, otherwise consult with one of the advisors) by passing a `tf.GPUOptions` as part of the optional `config` argument:

```python
# We are starting the event, with 50% of the GPU to each team

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
```

To check how occupied your allocated GPU is, use `watch nvidia-smi`. 
If the GPUs are too occupied we will find a solution so that each team get to train their network.


##### Docker
Docker is installed on the HPC, and will be used during the event. On this <a href="">link</a> you can fin NTNUs Docker guide. For more documentation see <a href="https://docs.docker.com/">https://docs.docker.com/</a>. Advisors from the AI Lab and Telenor will be guiding you, if needed. Please use `Dockerfile` in this repository as a template.  

##### Connecting to the server

Use the command below with your assigned user and password to connect to the server:

```bash
$ ssh <user>@<anakin-machine>.idi.ntnu.no
```

##### Relevant tutorials

###### *TBD*

#### Files of interest

* `Dockerfile`
* *More TBD*


#### Categories for evaluation

* **Most innovative idea**
* **Best use of AI on the data (method/application)**


#### Computational power
For processing we are using NTNUs new HPCs; *anakin01 and anakin02*:

* *GPU*: 2x NVIDIA Tesla V100 32GB
* *CPU*: 2x Intel Xeon Gold 6132 2.6G, 14C/28T, 10.4GT/s 2UPI
* *RAM*: 24x 32GB = 768GB


### Jury

* Kerstin Bach
* Massimiliano Ruocco
* Hai Thanh Nguyen
* Arjun Chandra
