# Open Code Checklist
To ensure that the code we publish is appropraite, clear and reusable we require a set way of working when pushing code to private and public repoistories.  It is difficult to set any precise but generic standards and so we ask that the following checklist is considered with mandatroy items filled out. 

## When publishing your code you need to make sure:
  
### you’re clear about who owns the code and how others can use it

- [X] Does your code have an appropriate licence and copyright notice?  (**Mandatory**)
- [X] Is there a README and does it document intended purpose? (**Mandatory**)
- [X] Is the README clear and concise? (**Optional** - use analytics unit template or [example](https://github.com/othneildrew/Best-README-Template/blob/master/BLANK_README.md))
- [No] Do you need to consider MHRA 'software as a medical device' guidance? (**Mandatory** - use [flowchart](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/999908/Software_flow_chart_Ed_1-08b-IVD.pdf))
- [X] Who has responsibility for ongoing support and communications for the code? (**Optional** Best practice to assign, else state the code is not maintained and when the last update was)
- [X] Has a responsible disclosure process for security issues been defined? (**Mandatory** - could use github issues)
- [See Contact] Who will address identified issues and security concerns? (**Mandatory**. - assign monitoring of above disclosure process)
- [X] Has semantic versioning been used? (**Optional**) 
- [X] Have [contribution guidelines](https://github.com/alphagov/govuk-frontend/blob/master/CONTRIBUTING.md) or [PR style guides](https://github.com/alphagov/styleguides/blob/master/pull-requests.md) been included? (**Optional**)
- [Yes - But not Versions] Are package dependencies and libaries documented with versions? (**Optional**)
- [n/a] Has the code been linked to any published outputs so others can easily link to the code? (**Optional**) 

### You do not release information that should remain closed

- [X] Does the code include any sensitive, personal, secret or top secret data/information? (**Mandatory**)
- [X] Does the code include any unreleased policy? (**Mandatory**)
- [X] Does the code include business sensitive algorithms (e.g. finance allocations)? (**Mandatory**)
- [X] Has written permission been obtained for any data stored from the data owner? (**Mandatory**)
- [X] Are owners of services which the code fulfils aware of the release? (**Mandatory**)
- [X] Are any data transfers conducted safely and securely? (**Mandatory**)
- [X] Are any credentials contained in the source code? (**Mandatory** - check in both current version and git history)
- [X] Are any secret keys contained in the source code? (**Mandatory** - check in both current version and git history)
- [X] Are any SQL server addresses or connection strings in the source code? (**Mandatory** - check in both current version and git history)
- [X] Are the commit messages informative? (**Optional**) 
- [X] Do the commit messages include any sensitive information (e.g. names)? (**Mandatory**)
- [X] Does the git history contain any sensitive information (e.g. at one time real data or credentials were in the code but have since been removed) (**Mandatory**)
- [X] Have notebook outputs been removed/checked for sensitive information? (**Mandatory** - check but some appropriate outputs maybe useful: [Example]( https://github.com/best-practice-and-impact/govcookiecutter/blob/main/%7B%7B%20cookiecutter.repo_name%20%7D%7D/.pre-commit-config.yaml))
- [X] Is configuration written as code and separated from analytical code? (**Optional**) 
- [X] have you checked any screenshots or figures in your outputs and documentation for information that shouldn't be released? (**Mandatory**)

### You store it in a repository managed by your department (to make licensing/copyright clear)

- [X] Is the code version controlled using GIT or similar? (**Optional**)
- [X] Is the code stored in your organisational GitHub account? Is it the same organisation that funds the relevant staff time? (**Optional**)

### Any third-party tools you use to host or manage your code follow the National Cyber Security Centre’s cloud security guidance

- [X] Are third party tools used within the code? (**Mandatory** check. Best practice is to keep an inventory)
- [X] If so do they adhere to the NCSC's [Cloud Security Principles](https://www.ncsc.gov.uk/collection/cloud-security/implementing-the-cloud-security-principles)? (**Mandatory**)

### An internal code review has been completed

- [ ] Has a colleague reviewed the code for sensitive data content and security vulnerabilities? (**Mandatory** - Best practice is to record automated code quality and security tools used)
- [ ] Has a code quality review been completed focussing on the end usability and clarity? (**Optional** - consider runing through the [example](https://best-practice-and-impact.github.io/qa-of-code-guidance/checklist_higher.html) or similar code quality checklist)
- [ ] Has the code been assessed for its [level or RAP](https://github.com/NHSDigital/rap-community-of-practice/blob/main/what_is_RAP/levels_of_RAP.md)(Reproducible Analytical Pipeline)? (**Optional**)
- [ ] Has the code undergone some level of testing.  The level of teting required will depend on the specific code and use-case but as minimum it should work in a fresh environment with arteficial data. (**Optional**)
