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

This repository holds a single notebook which contains a toy simulation of patient pathways to be optimised for both patient outcomes and system cost.  The simulation includes:
- Patient Class (incl. age, sex, diseases, clinical values, priority)
- Action Class (incl. capacity, effects of the action on the patient clincial values, cost, duration, queue)
- Pathway Class (incl. valid transitions and thresholds)
    - The next_action method chooses from the valid transitions using the demographic, clinical value and random thresholds
- Simulation Setup (default set to 10 patients, 10 pathways, 10 actions, 30 time steps)
- Q-Learning Setup (based on the action cost, clinical penalty and length of time queuing)
- Simulation Run 
- Visualisations of outcomes

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

#### Outputs
Visualisations of the system to be generated as well as figures showing usage, queues and outcomes.   The random seeds can be uncommented in the first cell to create reproducible results. 

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
