<a name="readme-top"></a>
[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)

<!-- page header -->
<br/>
<div align="center">
    <h2 align="center">SynPath - Toy Model</h2>
    <h3 align="center">Data Science and Applied AI Team, Data & Analytics (D&A)</h3>
    <p align="center">
        This project is in development, and should not be relied upon. Please do not use any code without consultation of the Data Science and Applied AI team (D&A). 
        <br/>
        <a href="https://github.com/nhsengland/SynPathToy/issues">Report Bug</a>
        <a href="https://github.com/nhsengland/SynPathToy">Request Feature</a>
    </p>
</div>

### About the Project

This repository holds python code for a toy simulation of patient pathways to be optimised for both patient outcomes and system cost.  

<img width="1002" height="730" alt="image" src="https://github.com/user-attachments/assets/4268d811-d450-49ad-bb02-7415c34e05bc" />

*Figure 1: Schematic of the simulation showing a patient with attributes interacting with multiple randomly assigned pathways which each have randomly assigned overlapping actions*

_**Note:** Only public or fake data are shared in this repository._

### Built With

[![Python v3.9](https://img.shields.io/badge/python-v3.9-blue.svg)](https://www.python.org/downloads/release/python-3916/)
- Core Python packages (e.g. numpy, pandas)
- Core python visulisation packages (e.g. matplotlib, networkx, seaborn)
- [heapq](https://docs.python.org/3/library/heapq.html) for the priority queue implementation

#### Data
As this is a toy simulation, fake data is generated within the code using random numbers.   It does not reflect any real world data or insights. 

### Usage
This code has been created to demonstrate some of the features required when optimising patient pathways.   It is to be used as a demonstration only. 

The code structure contained in '**project**' includes:
- 'main.py'
- 'config.py' Config (default set to 10 patients, 10 pathways, 10 actions, 30 time steps) 
    - At 10 patients this takes less than 1 second to run.   At 1,000 patients this takes 1hr30.
    - 'patient.py' Patient Class (incl. age, sex, diseases, comorbidities, clinical values, sickness, outcomes)
    - The progress_disease method simulates disease occurence over time
    - The clinical_decay method simulates the patient getting sicker over time
    - The apply_action methods simulates the patient getting better due to an activity
    - The score_outcomes method calculates the queue and clinical penalities for an individual
- 'action.py' Action Class (incl. capacity, effects of the action on the patient clincial values, cost, duration, queue)
    - The update_capacity allows some dynamic changes in supply
    - The assign method deals with how the individuals are placed in a queue for the activity using heapq
    - The execute method 
- 'pathway.py' Pathway Class (incl. valid transitions and thresholds)
    - The next_action method chooses from the valid transitions using q-learning to decide for each pathway which action to take dependent on the patient age-group and sickness and optimise over cost, benefit to the patient, patient queue time, and system overall queue length. 
- 'build.py' Simulation Build (Creates a set of random actions randomly connected with transistion and threshold matrices to define possible links)
- 'run.py' Simulation Run 
    - Runs for two major time steps to compare raw versus learnt systems
    - Runs for a user set number of steps
    - For each patient look at each pathway if the patient is on this pathway choose a next action for them to be added to the queue for.  Calculate the outcomes and log the activity.
- 'vis.py' Visualisations of outcomes

The '**Experiments**' Folder contains a full and minimium working example of the code as notebooks.

#### Outputs
Visualisations of the system to be generated as well as figures showing usage, queues and outcomes - saved on outputs/.   The random seeds can be uncommented in the first cell to create reproducible results. 

### Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### Contact
datascience@nhs.net

### License
Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
